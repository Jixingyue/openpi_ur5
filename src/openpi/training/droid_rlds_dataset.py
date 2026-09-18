"""
基于 RLDS 的 DROID 数据加载器。
openpi 通常使用 LeRobot 的数据加载器，但对于像 DROID 这样较大的数据集，它目前还不夠具伸缩性。
因此这里提供一个使用 RLDS 数据格式的数据加载器示例。
该数据加载器还会应用一些 DROID 专用的数据过滤 / 变换。
"""

from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):
    """DROID 数据集的动作空间。"""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,  # 强制使用仅关键字参数
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # 我们默认使用关节位置动作，因为它们允许在仿真中评估策略。
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # 如果你内存不够用，请调低此值，但要小心——低于约 100k 时打乱就不够随机了。
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- 一个 hack，以便不在顶层导入 tf
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- 一个 hack，以便不在顶层导入 tf
    ):
        # 在此处导入 tensorflow，以免在使用 RLDS 数据加载器时它不是必需的。
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds

        # 配置 Tensorflow 使用*无 GPU 设备*（以防与 PyTorch / JAX 冲突）
        tf.config.set_visible_devices([], "GPU")

        # 确保数据集权重之和为 1.0
        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        def prepare_single_dataset(dataset_cfg: RLDSDataset):
            # ds_name, version = dataset_name.split(":")
            ds_name, version = dataset_cfg.name, dataset_cfg.version
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )

            # 过滤掉任何不成功的轨迹——我们使用文件名来检查这一点
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # 重复数据集，因此我们永远不会缺数据。
            dataset = dataset.repeat()

            # 如果提供了过滤字典则加载它。
            # 过滤字典是一个 JSON 文件，将 episode 键映射到要采样的帧范围
            # （例如，
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # 表示保留第 0-99 帧和第 200-299 帧）。

            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)
                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []

                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            frame_key = f"{episode_key}--{t}"
                            keys_tensor.append(frame_key)
                            values_tensor.append(True)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            def restructure(traj):
                """重新格式化观测和动作键，并采样语言指令。"""
                # 重要：我们使用关节*位置*动作空间——更易于仿真！
                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                            if action_space == DroidActionSpace.JOINT_POSITION
                            else traj["action_dict"]["joint_velocity"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )
                # 训练时从 DROID 的两个外部图像中随机采样一个（我们一次只使用一个进行训练）。
                # 注意：“left”指的是立体图像对中的左相机，我们只在左相机上训练。
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]
                # 从三个语言指令中随机采样一个
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                traj_len = tf.shape(traj["action"])[0]
                indices = tf.as_string(tf.range(traj_len))

                # 数据过滤：
                # 通过拼接录制文件夹路径、文件路径以及每个 step 的时间步索引，计算出一个唯一标识的 step ID。
                # 它将用于索引过滤哈希表，如果返回 true，
                # 则该帧通过了过滤。
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                passes_filter = self.filter_table.lookup(step_id)

                return {
                    "actions": actions,
                    "observation": {
                        "image": exterior_img,
                        "wrist_image": wrist_img,
                        "joint_position": traj["observation"]["joint_position"],
                        "gripper_position": traj["observation"]["gripper_position"],
                    },
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """将 episode 拆分为动作 chunk。"""
                traj_len = tf.shape(traj["actions"])[0]

                # 对轨迹中的每一步，构造接下来 n 个动作的索引
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # 限制到序列长度 --> 最后的 chunk 会重复最后一个动作
                # 这是合理的，因为我们使用的是绝对的关节 + 夹爪位置动作
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # 为每个 chunk 收集动作
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # 展平：从轨迹数据集映射为单个动作 chunk 的数据集
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # 过滤掉未通过过滤的数据
            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)

            # 从输出中移除 "passes_filter" 键
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)

            # 解码图像：RLDS 保存的是编码后的图像，仅为了效率而现在才解码
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)
        all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)
        # Note =>> 似乎能在不影响速度的情况下减少内存使用？
        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # 这是过滤后 DROID 中样本的近似数量。
        # 硬编码比遍历数据集计算更简单。
        return 20_000_000
