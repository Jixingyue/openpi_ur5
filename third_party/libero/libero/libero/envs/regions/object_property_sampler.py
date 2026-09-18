import collections
import numpy as np
import os
import robosuite
import xml.etree.ElementTree as ET

from copy import copy

from robosuite.models.objects import MujocoObject


class ObjectPropertySampler:
    """
    物体放置采样器的基类。
    参数:
        name (str): 此采样器的名称。
        mujoco_objects (None or MujocoObject or list of MujocoObject): 单个模型或MJCF对象模型列表
        ensure_object_boundary_in_range (bool): 如果为True，将确保物体在给定边界内
            （应由子类实现）
        ensure_valid_placement (bool): 如果为True，将检查正确（有效）的物体放置
        reference_pos (3-array): 采样将相对于此全局(x,y,z)位置进行
        z_offset (float): 为放置添加小的z偏移。这对于不移动的固定物体很有用
            （即没有自由关节）以将它们放置在桌面上方。
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
    ):
        # 设置属性
        self.name = name
        if mujoco_objects is None:
            self.mujoco_objects = []
        else:
            # 浅拷贝列表，这样我们不会修改输入的列表但仍然保留对象引用
            self.mujoco_objects = (
                [mujoco_objects]
                if isinstance(mujoco_objects, MujocoObject)
                else copy(mujoco_objects)
            )

    def add_objects(self, mujoco_objects):
        """
        向此采样器添加额外物体。检查确保没有已存储的相同物体。
        参数:
            mujoco_objects (MujocoObject or list of MujocoObject): 单个模型或MJCF对象模型列表
        """
        mujoco_objects = (
            [mujoco_objects]
            if isinstance(mujoco_objects, MujocoObject)
            else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), "Object '{}' already in sampler!".format(obj.name)
            self.mujoco_objects.append(obj)

    def reset(self):
        """
        重置此采样器。从此采样器中移除所有mujoco物体。
        """
        self.mujoco_objects = []

    def sample(self, predicate_name=None):
        """
        在表面上均匀采样（不一定是桌面）。
        参数:
            fixtures (dict): 当前场景中物体放置的字典，以及不应与新采样物体接触的任何其他相关
                障碍物。用于确保新生成的放置是有效的。应该是物体名称映射到(pos, quat, MujocoObject)
            reference (str or 3-tuple or None): 如果提供，采样相对放置。可以是字符串，
                对应@fixtures中找到的现有物体，或直接的(x,y,z)值。如果为None，将相对于
                此采样器的`'reference_pos'`值进行采样。
            on_top (bool): 如果为True，在参考物体顶部采样放置。
        返回:
            dict: 所有物体放置的字典，将object_names映射到(pos, quat, obj)，包括
                @fixtures中指定的放置。注意quat为(w,x,y,z)格式
        """
        raise NotImplementedError


class OpenCloseSampler(ObjectPropertySampler):
    def __init__(
        self,
        name,
        state_type,
        mujoco_objects=None,
        joint_ranges=(0.0, 0.0),
    ):
        assert state_type in ["open", "close"]
        self.state_type = state_type
        self.joint_ranges = joint_ranges
        assert self.joint_ranges[0] <= self.joint_ranges[1]
        super().__init__(name, mujoco_objects)

    def sample(self):
        return np.random.uniform(high=self.joint_ranges[1], low=self.joint_ranges[0])


class TurnOnOffSampler(ObjectPropertySampler):
    def __init__(
        self,
        name,
        state_type,
        mujoco_objects=None,
        joint_ranges=(0.0, 0.0),
    ):
        assert state_type in ["turnon", "turnoff"]
        self.state_type = state_type
        self.joint_ranges = joint_ranges
        assert self.joint_ranges[0] <= self.joint_ranges[1]
        super().__init__(name, mujoco_objects)

    def sample(self):
        return np.random.uniform(high=self.joint_ranges[1], low=self.joint_ranges[0])
