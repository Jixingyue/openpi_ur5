# UR5 Example

Below we provide an outline of how to implement the key components mentioned in the "Finetune on your data" section of the [README](../README.md) for finetuning on UR5 datasets.

First, we will define the `UR5Inputs` and `UR5Outputs` classes, which map the UR5 environment to the model and vice versa. Check the corresponding files in `src/openpi/policies/libero_policy.py` for comments explaining each line.

```python

@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # First, concatenate the joints and gripper into the state vector.
        state = np.concatenate([data["joints"], data["gripper"]])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        # Create inputs dict.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        # Since the robot has 7 action dimensions (6 DoF + gripper), return the first 7 dims
        return {"actions": np.asarray(data["actions"][:, :7])}

```

Next, we will define the `UR5DataConfig` class, which defines how to process raw UR5 data from LeRobot dataset for training. For a full example, see the `LeRobotLiberoDataConfig` config in the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py).

```python

@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[UR5Inputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

```

Finally, we define the TrainConfig for our UR5 dataset. Here, we define a config for fine-tuning pi0 on our UR5 dataset. See the [training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py) for more examples, e.g. for pi0-FAST or for LoRA fine-tuning.

```python
TrainConfig(
    name="pi0_ur5",
    model=pi0.Pi0Config(),
    data=LeRobotUR5DataConfig(
        repo_id="your_username/ur5_dataset",
        # This config lets us reload the UR5 normalization stats from the base model checkpoint.
        # Reloading normalization stats can help transfer pre-trained models to new environments.
        # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
        base_config=DataConfig(
            # This flag determines whether we load the prompt (i.e. the task instruction) from the
            # ``task`` field in the LeRobot dataset. The recommended setting is True.
            prompt_from_task=True,
        ),
    ),
    # Load the pi0 base model checkpoint.
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)
```


## 数据与训练、推理、真机控制（按顺序执行）

以下命令默认在 **openpi 仓库根目录**执行（先 `cd /path/to/openpi`）。LeRobot 数据集目录名为 **`data_converted`** 时，其父目录需设为 **`HF_LEROBOT_HOME`**（例如数据在 `/home/lab/extra/data_converted` 则：`export HF_LEROBOT_HOME=/home/lab/extra`）。

若 `compute_norm_stats` 或训练读 parquet 时出现 **`Feature type 'List' not found`**，请先对数据集执行一次元数据修复（仅需一次）：

```bash
export HF_LEROBOT_HOME=/path/to/parent_of_dataset_folder
uv run scripts/fix_lerobot_parquet_hf_metadata.py "$HF_LEROBOT_HOME/data_converted"
```

---

### 1. 计算归一化统计（norm stats，含均值 / 方差 / 分位数等）

在训练前必须执行；输出写入 `assets/pi05_ur5_single_cam/data_converted/`（详见根目录 [README](../README.md)）。

```bash
export HF_LEROBOT_HOME=/path/to/parent_of_dataset_folder
uv run scripts/compute_norm_stats.py --config-name pi05_ur5_single_cam
```

---

### 2. 微调 π₀.₅（JAX，`pi05_ur5_single_cam`）

将 `--exp-name` 换成你的实验名；与 `--overwrite` 二选一可用 **`--resume`** 接续训练（勿同时使用）。

```bash
export HF_LEROBOT_HOME=/path/to/parent_of_dataset_folder
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_ur5_single_cam --exp-name=my_ur5_run --overwrite
```

---

### 3. 启动推理服务（策略 WebSocket）

将 `CHECKPOINT` 换成你的 checkpoint 路径（例如 `checkpoints/pi05_ur5_single_cam/my_ur5_run/10000`）；若做零样本试验可指向官方 **`gs://openpi-assets/checkpoints/pi05_base`**（效果取决于任务）。

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_ur5_single_cam \
  --policy.dir CHECKPOINT
```

---

### 4. 启动真机控制程序（UR5e + D435i 客户端）

部署前务必根据现场修改 **`examples/ur5/constants.py`** 中的 **`HOME_Q6`**。将 `<策略机IP>`、`<UR控制器IP>`、`--prompt` 换成实际值。

先安装本目录依赖（**Robotiq 串口控制**需要其中的 **`pymodbus`**；仅用机械臂无夹爪下发时也可只装该文件所列依赖）：

```bash
cd /path/to/openpi
uv pip install -r examples/ur5/requirements.txt   # 含 pymodbus
```

**仅机械臂（RTDE 6 轴伺服；第 7 维夹爪观测仍可用 RTDE 寄存器或 `--gripper-obs-default`，不向夹爪硬件发运动）**：

```bash
cd /path/to/openpi
uv run python -m examples.ur5.main \
  --host <策略机IP> --port 8000 \
  --robot-ip <UR控制器IP> \
  --prompt "与你的训练任务一致的指令" \
  --action-horizon 10 \
  --max-hz 30
```

**Robotiq 2F-85（Modbus RTU，与 `main_ur3_all.py` 类似）**：启用 **`--robotiq-port`** 后，第 7 维 `state` 从夹爪读回，策略输出的第 7 维会映射为开合指令（默认 `--robotiq-open-pos` / `--robotiq-close-pos` 与采集脚本常用值一致，可按数据标定调整）。

```bash
cd /path/to/openpi
uv pip install -r examples/ur5/requirements.txt   # 含 pymodbus
uv run python -m examples.ur5.main \
  --host <策略机IP> --port 8000 \
  --robot-ip <UR控制器IP> \
  --robotiq-port /dev/ttyUSB0 \
  --robotiq-slave-id 9 \
  --prompt "与你的训练任务一致的指令" \
  --action-horizon 10 \
  --max-hz 30
```

若未使用 `uv run` 在仓库根目录执行，需将仓库根目录加入 **`PYTHONPATH`**，以便解析 `examples` 包。

---

## UR5e 真机部署（D435i + RTDE + 远程策略）

本目录提供与 `examples/aloha_real` 相同思路的客户端：**本机采图像与关节状态 → WebSocket 发往 openpi 策略服务 → 接收动作块 → RTDE 下发**。观测字段与训练配置 **`pi05_ur5_single_cam`** 对齐：顶层 **`image`**（`uint8`、**`C,H,W`**）、**`state`**（**7** 维：`6` 关节弧度 + **`[0,1]`** 夹爪标量）、**`prompt`**。

**命令顺序**：归一化统计 → 微调 → 推理服务 → 真机客户端，见上文 **「数据与训练、推理、真机控制（按顺序执行）」**；本节仅补充依赖、参数说明与安全提示。

### 依赖

```bash
cd /path/to/openpi
uv pip install -e packages/openpi-client
uv pip install -r examples/ur5/requirements.txt
```

需要：UR 控制器开启 **RTDE**；D435i 使用 **Intel RealSense** 驱动；机器人程序中允许外部 RTDE 控制（与现场安全规程一致）。

### 客户端参数说明

- **`--dry-run`**：只连策略服务器与相机，不连 RTDE 发运动（调试用）；**`--robotiq-port` 会被忽略**（不占串口）。
- **`--use-fake-camera`**：不用 RealSense（黑图），仅测策略链路。
- **`--gripper-obs-register`**：若夹爪开合由 URCap 写到 **RTDE output double register**，填寄存器编号，则第 7 维 `state` 从该寄存器读（并 `clip` 到 `[0,1]`）；默认 **`-1`** 表示用 **`--gripper-obs-default`** 常数（与数据集中常开夹爪近似时可临时用，**闭环精度差**）。若同时指定 **`--robotiq-port`**，第 7 维优先从 **Robotiq `get_status()`** 读回。
- **夹爪执行**：指定 **`--robotiq-port`** 时，通过 **`examples/ur5/robotiq_2f85.py`** 经 Modbus RTU 下发 **`move(position, speed, force)`**（可用 **`--robotiq-speed` / `--robotiq-force` / `--robotiq-open-pos` / `--robotiq-close-pos`** 等调参）。未指定时第 7 维动作**不下发**到夹爪硬件（仅用 RTDE 控 6 轴）。

### 安全提示

在修改 **`HOME_Q6`**、首次上电、以及接入真实夹爪控制之前，请使用 **`--dry-run`** / **`--use-fake-camera`** 验证网络与策略；现场需满足急停、围栏与 UR 安全平面等企业规范。本示例代码仅供研究参考，作者不对误用造成的设备或人身伤害负责。




