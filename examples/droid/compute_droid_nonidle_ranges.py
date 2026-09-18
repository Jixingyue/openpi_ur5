"""
遍历 DROID 数据集，创建一个 json 映射，将 episode 唯一 ID 映射到应在训练中采样的时间步范围
（其余时间步将被过滤掉）。

过滤逻辑：
我们查找至多包含 min_idle_len 个连续空闲帧的连续步长范围
（默认为 7 —— 由于大多数 DROID action-chunking 策略会执行每个 chunk 中生成的前 8 个动作，
因此这样过滤意味着策略不会卡在输出静止动作上）。另外，我们也仅保留长度至少为 min_non_idle_len
（默认为 16 帧 = ~1 秒）的非空闲范围，同时从每个范围的末尾删除 filter_last_n_in_ranges 帧
（因为它们都对应于包含很多空闲动作的 action chunk）。

这会留下由连续且有意义的运动组成的轨迹片段。在这个过滤后的集合上训练
会产生输出更少静止动作的策略（即更少地“卡”在某些状态）。
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 设置为你想使用的 GPU，或留空以使用 CPU

builder = tfds.builder_from_directory(
    # `droid` 目录的路径（而非其父目录）
    builder_dir="<path_to_droid_dataset_tfds_files>",
)
ds = builder.as_dataset(split="train", shuffle_files=False)
tf.data.experimental.ignore_errors(ds)

keep_ranges_path = "<path_to_where_to_save_the_json>"

min_idle_len = 7  # 若连续空闲帧数超过此值，则全部过滤掉
min_non_idle_len = 16  # 若连续非空闲帧数少于此值，则全部过滤掉
filter_last_n_in_ranges = 10  # 使用过滤字典时，从每个范围末尾删除的帧数

keep_ranges_map = {}
if Path(keep_ranges_path).exists():
    with Path(keep_ranges_path).open("r") as f:
        keep_ranges_map = json.load(f)
    print(f"Resuming from {len(keep_ranges_map)} episodes already processed")

for ep_idx, ep in enumerate(tqdm(ds)):
    recording_folderpath = ep["episode_metadata"]["recording_folderpath"].numpy().decode()
    file_path = ep["episode_metadata"]["file_path"].numpy().decode()

    key = f"{recording_folderpath}--{file_path}"
    if key in keep_ranges_map:
        continue

    joint_velocities = [step["action_dict"]["joint_velocity"].numpy() for step in ep["steps"]]
    joint_velocities = np.array(joint_velocities)

    is_idle_array = np.hstack(
        [np.array([False]), np.all(np.abs(joint_velocities[1:] - joint_velocities[:-1]) < 1e-3, axis=1)]
    )

    # 查找哪些步从空闲转为非空闲，反之亦然
    is_idle_padded = np.concatenate(
        [[False], is_idle_array, [False]]
    )  # 首尾都为 False，以便首步的空闲被视为运动的开始

    is_idle_diff = np.diff(is_idle_padded.astype(int))
    is_idle_true_starts = np.where(is_idle_diff == 1)[0]  # +1 转变 --> 从空闲转为非空闲
    is_idle_true_ends = np.where(is_idle_diff == -1)[0]  # -1 转变 --> 从非空闲转为空闲

    # 查找哪些步对应长度至少为 min_idle_len 的空闲段
    true_segment_masks = (is_idle_true_ends - is_idle_true_starts) >= min_idle_len
    is_idle_true_starts = is_idle_true_starts[true_segment_masks]
    is_idle_true_ends = is_idle_true_ends[true_segment_masks]

    keep_mask = np.ones(len(joint_velocities), dtype=bool)
    for start, end in zip(is_idle_true_starts, is_idle_true_ends, strict=True):
        keep_mask[start:end] = False

    # 获取所有长度至少为 16 的非空闲范围
    # 与上面逻辑相同，但针对 keep_mask，以便我们过滤掉长度 < min_non_idle_len 的连续范围
    keep_padded = np.concatenate([[False], keep_mask, [False]])

    keep_diff = np.diff(keep_padded.astype(int))
    keep_true_starts = np.where(keep_diff == 1)[0]  # +1 转变 --> 从过滤转为保留
    keep_true_ends = np.where(keep_diff == -1)[0]  # -1 转变 --> 从保留转为过滤

    # 查找哪些步对应长度至少为 min_non_idle_len 的非空闲段
    true_segment_masks = (keep_true_ends - keep_true_starts) >= min_non_idle_len
    keep_true_starts = keep_true_starts[true_segment_masks]
    keep_true_ends = keep_true_ends[true_segment_masks]

    # 添加从 episode 唯一 ID 键到要保留的非空闲范围列表的映射
    keep_ranges_map[key] = []
    for start, end in zip(keep_true_starts, keep_true_ends, strict=True):
        keep_ranges_map[key].append((int(start), int(end) - filter_last_n_in_ranges))

    if ep_idx % 1000 == 0:
        with Path(keep_ranges_path).open("w") as f:
            json.dump(keep_ranges_map, f)

print("Done!")
with Path(keep_ranges_path).open("w") as f:
    json.dump(keep_ranges_map, f)
