import copy

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from PIL import Image
from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import Dataset

"""
    来自 Robomimic 的辅助函数，用于将 hdf5 演示读取为序列数据集

    问题：robomimic 的 SequenceDataset 有两个属性：seq_len 和 frame_stack，
    原则上我们应该使用 seq_len，但两者的填充方式不同。
    这就是为什么我们目前使用 frame_stack 而不是 seq_len。
"""


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # 长度为 10 的时间序列
        pad_frame_stack=True,
        pad_seq_length=True,  # 填充每条轨迹的最后一个观测值，以确保所有序列都能被采样
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # 将数据集缓存在内存中以避免重复的文件 i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # 可以在这里可选地提供一个过滤键
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # 创建一个映射，将 dataloader 的当前 idx 映射到原始任务数据 idx
        # 假设我们有任务 1,2,3，大小为 3,5,4，那么 idx 看起来如下
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # 这样做，当我们拼接数据集时，每个任务都将拥有相等数量的演示
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)
