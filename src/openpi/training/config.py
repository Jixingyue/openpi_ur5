"""可用配置的列表请参见 _CONFIGS。"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.ur5_policy as ur5_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# 规避一个 tyro 问题：直接使用 nnx.filterlib.Filter。
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """决定资源（例如归一化统计量）的位置，它们将用于搭建数据流水线。

    这些资源会被复制到 checkpoint 内部的 `assets/asset_id` 目录下。

    这可以用于从另一个 checkpoint（例如基础模型 checkpoint）或某个其他中心化的位置
    加载资源。例如，要在微调期间从基础模型 checkpoint 加载 Trossen 机器人的归一化统计量，使用：

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # 资源目录。若未提供，则使用配置的 assets_dirs。这对于从另一个
    # checkpoint（例如基础模型 checkpoint）或某个其他中心化位置加载资源很有用。
    assets_dir: str | None = None

    # 资源 id。若未提供，则使用 repo id。这允许用户引用描述不同机器人平台的
    # 资源。
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot 仓库 id。若为 None，将创建假数据。
    repo_id: str | None = None
    # 资源目录中包含数据资产的子目录。
    asset_id: str | None = None
    # 包含预先计算好的归一化统计量。若为 None，将不执行归一化。
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 用于将输入从数据集特定的格式适配为数据变换所期望的
    # 通用格式。
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 数据变换，通常包含机器人特定的变换。将在数据被归一化
    # 之前应用。参见 `model.Observation` 和 `model.Actions` 了解归一化后的
    # 数据。
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 模型特定的变换。将在数据被归一化之后应用。
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 若为 true，将使用分位数归一化。否则使用常规的 z-score 归一化。
    use_quantile_norm: bool = False

    # 数据加载器用于生成动作序列的键名。序列的长度由模型配置中的
    # `action_horizon` 字段定义。如果你的 LeRobot 数据集使用不同的键来表示
    # 动作，则需要调整此项。
    action_sequence_keys: Sequence[str] = ("actions",)

    # 若为 true，将使用 LeRobot 数据集的 task 来定义 prompt。
    prompt_from_task: bool = False

    # 仅用于 RLDS 数据加载器（即目前仅用于 DROID）。
    rlds_data_dir: str | None = None
    # DROID 数据集的动作空间。
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # 要从中采样的数据集列表：name、version、weight，以及可选的 filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """创建一个 group。"""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """为标准 pi0 模型创建模型变换。"""

    # 若提供，将决定模型使用的默认 prompt。
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # LeRobot 仓库 id。
    repo_id: str = tyro.MISSING
    # 决定资源将如何被加载。
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # 将由工厂更新的基础配置。
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """创建数据配置。"""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # 数据变换的工厂。
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # 模型变换的工厂。
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # 若为 true，将在传递给模型之前，将关节维度转换为相对于当前状态的增量。
    # 夹爪维度将保持为绝对值。
    use_delta_joint_actions: bool = True
    # 若提供，当输入数据中不存在 "prompt" 键时，它将被注入输入数据。
    default_prompt: str | None = None
    # 若为 true，将会将关节和夹爪的数值从标准 Aloha 空间转换到 pi 内部运
    # 行时所使用、用于训练基础模型的空间。使用标准 Aloha 数据的人应该将其设为 true。
    adapt_to_pi: bool = True

    # Repack 变换。
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # 将从数据集中读取动作序列时使用的动作键。
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    该配置用于配置在数据流水线各环节应用的变换。对于你自己的数据集，
    你可以拷贝该类，并根据下面的注释修改变换以匹配你的数据集。
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # repack 变换*仅*应用于来自数据集的数据，而在推理时*不*应用。我们可以用它
        # 使来自数据集的输入尽可能与来自推理环境的输入保持一致（例如匹配键名）。
        # 下面，我们将数据集中的键（我们在数据转换脚本中定义的）与我们在推理流水线
        # 中使用的键（在 libero 的推理脚本中定义）进行匹配。对于你自己的数据集，先弄清楚你的
        # 环境传给策略服务器的键有哪些，然后修改下面的映射，使你数据集的键能匹配到那些目标键。
        # repack 变换在这里只是重映射键名。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据变换既应用于来自数据集的数据，也应用于推理时。
        # 下面，我们定义输入模型的数据变换（``inputs``）以及输出模型的数据变换
        # （``outputs``）（后者仅在推理时使用）。我们在 `libero_policy.py` 中定义了这些变换。
        # 你可以查看那里详细的注释，了解如何修改变换以匹配你的数据集。一旦你创建了自己的变换，
        # 就可以用你自己的变换替换下面的变换。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 一个额外的数据变换：pi0 模型是在增量动作（相对于每个动作 chunk 中的首个
        # 状态）上训练的。如果你的数据是``绝对``动作（例如目标关节角度），
        # 你可以取消下面这行的注释，将动作转换为增量动作。唯一的例外
        # 是夹爪动作，它始终是绝对的。
        # 在下面的例子中，我们会将增量转换应用于前 6 个动作（关节），而
        # 对第 7 个动作（夹爪）保持不变，即绝对值。
        # 在 Libero 中，数据集中的原始动作已经是增量动作，所以我们*不*需要
        # 应用单独的增量转换（这也是它被注释掉的原因）。根据你的数据集是否默认使用
        # ``绝对``或``增量``动作，来选择是否应用此变换。

        # LIBERO 已经将动作表示为增量，但我们有一些旧的 Pi0 checkpoint 是用这个额外的增量变换训练的。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # 模型变换包括诸如将 prompt 和动作目标 tokenize 等操作
        # 对于你自己的数据集，这里无需修改任何内容。
        model_transforms = ModelTransformFactory()(model_config)

        # 我们为训练和推理返回所有数据变换。这里无需修改任何内容。
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    """LeRobot 数据集 → UR5 策略格式。参见 `examples/ur5/README.md`。

    预期的**展平**数据集键（repack 之前），与默认的 `RepackTransform` 相匹配：
    `image`、`wrist_image`、`joints`、`gripper`、`action`，以及 `prompt`（或者依赖 `prompt_from_task`）。
    如果你的 Hub 数据集使用了不同的路径（例如 `observation.images.cam_high`），请编辑 `create` 中的 repack
    字典，并将 `DataConfig.action_sequence_keys` 设置为你动作列的名称。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur5_policy.UR5Inputs(model_type=model_config.model_type)],
            outputs=[ur5_policy.UR5Outputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUR5SingleCamDataConfig(DataConfigFactory):
    """面向本地 / Hub 数据集的 LeRobot v2 单相机 UR5（``image``、``state``、``actions``）。

    Repack 预期的展平键：``image``、``state``、``actions``、``prompt``（``prompt`` 可以
    来自 ``prompt_from_task``）。将 ``HF_LEROBOT_HOME`` 设为数据集目录的**上级**，以便
    ``repo_id`` 与文件夹名相匹配（例如 ``HF_LEROBOT_HOME=/home/lab/extra`` 且 ``repo_id=data_converted``）。
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur5_policy.UR5SingleCameraInputs(model_type=model_config.model_type)],
            outputs=[ur5_policy.UR5Outputs()],
        )

        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    用于在 DROID 上训练的配置，使用 RLDS 数据格式（以便在大数据集上进行高效训练）。
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # 过滤选项。可以传入一个字典的路径，将 episode 映射到时间步范围，
    # 用以表示要保留的时间步范围 (start, end)。Episode 可以通过
    # f"{recording_folderpath}--{file_path}" 唯一标识，两者都存在于 RLDS 的 episode 元数据中。

    # 要从中采样的数据集列表：name、version、weight，以及可选的 filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # 数据加载器返回绝对的关节位置动作——为训练将其转换为增量动作。
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    LeRobot 格式下自定义 DROID 数据集的示例数据配置。
    要将你的自定义 DROID 数据集（小于几十小时）转换为 LeRobot 格式，参见 examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # 我们假定是关节*速度*动作，因此我们*不*应该再应用额外的增量变换。
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 配置的名称。必须唯一。将用于引用此配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称。将用于命名元数据和 checkpoint 目录。
    exp_name: str = tyro.MISSING

    # 定义模型配置。部分属性（action_dim、action_horizon 和 max_token_len）由所有模型共享
    # ——参见 BaseModelConfig。具体的模型实现（例如 Pi0Config）继承自 BaseModelConfig，并可
    # 定义额外的属性。
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 权重加载器可以在模型初始化后选择性地（可能部分地）从磁盘加载权重。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 可选的 PyTorch checkpoint 路径，用于从中加载权重。
    pytorch_weight_path: str | None = None

    # PyTorch 训练的精度。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定哪些权重应该被冻结。
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 决定要训练的数据。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资源（例如归一化统计量）的基础目录。
    assets_base_dir: str = "./assets"
    # checkpoint 的基础目录。
    checkpoint_base_dir: str = "./checkpoints"

    # 训练期间随机生成器使用的随机种子。
    seed: int = 42
    # 全局批次大小。
    batch_size: int = 32
    # 数据加载器使用的工作进程数量。增加此数量会加快数据加载，但
    # 会增加内存和 CPU 使用量。
    num_workers: int = 2
    # 要运行的训练步数（批次数）。
    num_train_steps: int = 30_000

    # 多久（以步计）记录一次训练指标。
    log_interval: int = 100
    # 多久（以步计）保存一次 checkpoint。
    save_interval: int = 1000
    # 若设置，任何匹配的 step % keep_period == 0 的已有 checkpoint 将不会被删除。
    keep_period: int | None = 5000

    # 若为 true，当 checkpoint 目录已存在时会覆写它。
    overwrite: bool = False
    # 若为 true，将从最后一个 checkpoint 恢复训练。
    resume: bool = False

    # 若为 true，将启用 wandb 日志。
    wandb_enabled: bool = True

    # 用于向策略服务器传递元数据。
    policy_metadata: dict[str, Any] | None = None

    # 如果该值大于 1，将启用 FSDP 并在指定数量的设备间分片；总体
    # 设备内存会降低，但训练可能变慢。
    # 例如：若设备总数为 4 且 fsdp 设备数为 2；则模型会分片到 2 个设备，并在 2 组设备间
    # 运行数据并行。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """获取此配置的资源目录。"""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """获取此配置的 checkpoint 目录。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取可训练参数的过滤器。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 如果你在代码中需要按名称获取配置，请使用 `get_config`。
_CONFIGS = [
    #
    # 推理 Aloha 配置。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # 推理 DROID 配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # 微调 Libero 配置。
    #
    # 这些训练配置定义了在你自己的数据集上微调基础模型时使用的超参数。
    # 它们用于定义关键要素，例如你训练所用的数据集、使用的基础 checkpoint，
    # 以及诸如运行多少训练步数或使用多大学习率等其他超参数。
    # 对于你自己的数据集，你可以复制这个类，并根据下面的注释修改数据集名称和数据变换。
    TrainConfig(
        # 修改名称以反映你的模型和数据集。
        name="pi0_libero",
        # 这里你定义模型配置——在本例中我们使用 pi0 作为模型
        # 架构并执行*完整*微调。在下面的示例中，我们展示如何修改
        # 它以执行*低显存*（LORA）微调，并使用 pi0-FAST 作为替代架构。
        model=pi0_config.Pi0Config(),
        # 这里你定义训练所用的数据集。在本例中我们使用 Libero
        # 数据集。对于你自己的数据集，你可以修改 repo_id 指向你的数据集。
        # 同时修改 DataConfig，使用你上面为你的数据集新建的配置。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 此标志决定我们是否从 LeRobot 数据集中的
                # ``task`` 字段加载 prompt（即任务指令）。若设为 True，prompt 将出现在
                # 输入 dict 中一个名为 ``prompt`` 的字段里。推荐设置为 True。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 这里你定义想要加载哪个预训练 checkpoint 来初始化模型。
        # 它应当与你在上面选择的模型配置相匹配——即在本例中我们使用 pi0 基础模型。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 下面你可以定义其他超参数，例如学习率、训练步数等。
        # 查阅基础的 TrainConfig 类以获取所有可用超参数的完整列表。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 这里是一个加载 pi0 模型用于 LoRA 微调的示例。
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # freeze filter 定义了训练期间哪些参数应当被冻结。
        # 我们在模型配置中提供了一个便捷函数，用于返回给定模型配置
        # 在 LoRA 微调时的默认 freeze filter。只需确保它与你上面选择的模型配置
        # 相匹配即可。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # 这里是一个加载 pi0-FAST 模型用于完整微调的示例。
        # 修改 action_dim 和 action_horizon 以匹配你的数据集（action horizon 等于
        # 期望的动作块长度）。
        # max_token_len 是模型能够处理的最大（非图像）token 数量。
        # 它包括 token 化的 prompt、本体感知状态以及（FAST-tokenized）动作 token。
        # 该值选得太小可能会在序列末尾截断 token（代码会抛出一个
        # 警告），而选得太大则会浪费显存（因为我们将每个批次元素填充到
        # max_token_len）。一个经验法则是：单臂机器人使用约 180，双臂机器人使用约 250。
        # 通常这里先偏向取较小的值，如果你在训练期间看到大量警告，
        # 再酌情增大该值。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 注意这里我们加载的是 pi0-FAST 基础模型 checkpoint。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # 这里是一个加载 pi0-FAST 模型用于 LoRA 微调的示例。
        # 关于设置 action_dim、action_horizon 和 max_token_len，参见上面的注释。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 同样，在提取 freeze filter 时务必与上面的模型配置相匹配，
        # 该 filter 指定了 LoRA 微调期间哪些参数应当被冻结。
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # 微调 UR5（LeRobot）。将 `repo_id` 设为你的数据集；如有需要，对齐 `LeRobotUR5DataConfig` 中的 repack 键名。
    #
    TrainConfig(
        name="pi0_ur5",
        model=pi0_config.Pi0Config(),
        data=LeRobotUR5DataConfig(
            repo_id="your_username/ur5_dataset",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    #
    # 在本地 LeRobot UR5 上的 π₀.₅（LoRA）：单个固定相机，7D 状态 + 7D 动作。比完整微调显存占用更低。
    # 前提：export HF_LEROBOT_HOME=<数据集文件夹的父目录>（数据集目录名 == 下面的 repo_id）。
    #
    TrainConfig(
        name="pi05_ur5_single_cam",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotUR5SingleCamDataConfig(
            repo_id="data_converted",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
                action_sequence_keys=("actions",),
            ),
        ),
        batch_size=8,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        ema_decay=None,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
    ),
    #
    # 微调 Aloha 配置。
    #
    # 这是一个测试配置，用于说明如何在自定义 LeRobot 数据集上训练。
    # 关于如何转换并在你自己的 Aloha 数据集上训练的说明，参见 examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # 微调 DROID 配置。
    #
    TrainConfig(
        # 此配置用于在*完整* DROID 数据集上微调 pi0-FAST-base。
        # 我们使用 RLDS 数据加载，使在这个大型数据集上训练变得可行。
        # 关于在你自己的 DROID 数据集上微调，参见下文。
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 将其设为你的 DROID RLDS 数据集路径（即 `droid` 目录的父目录）。
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k 步应当已足够，在 8x H100 上约需 2 天
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # 重要：RLDS DataLoader 要求 num_workers=0，它在内部处理多进程
    ),
    TrainConfig(
        # 此配置用于在*完整* DROID 数据集上微调 pi05。
        # 我们使用 RLDS 数据加载，使在这个大型数据集上训练变得可行。
        # 关于在你自己的 DROID 数据集上微调，参见下文。
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # 将其设为你的 DROID RLDS 数据集路径（即 `droid` 目录的父目录）。
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # 重要：RLDS DataLoader 要求 num_workers=0，它在内部处理多进程
    ),
    TrainConfig(
        # 此配置用于在自定义（较小）的 DROID 数据集上微调 pi05-DROID。
        # 这里我们使用 LeRobot 数据格式（与所有其他微调示例一样）
        # 要将你的自定义 DROID 数据集（<几十小时）转换为 LeRobot 格式，参见 examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 使用 32 维动作进行训练
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # 替换为你的自定义 DROID LeRobot 数据集 repo id。
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # 重要：微调时请复用原始的 DROID 归一化统计量！
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim 配置。此配置用于演示如何在简单的仿真环境上训练。
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # 调试配置。
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena 与 PolaRiS 配置。
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """按名称获取一个配置。"""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
