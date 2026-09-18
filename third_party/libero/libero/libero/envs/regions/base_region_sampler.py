import collections
import numpy as np
import os
import robosuite
import xml.etree.ElementTree as ET

from copy import copy
from robosuite.utils.errors import RandomizationError
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.utils.transform_utils import quat_multiply
import robosuite.utils.transform_utils as T


class MultiRegionRandomSampler(ObjectPositionSampler):
    """
    将所有物体在桌面内均匀随机放置。
    参数:
        name (str): 此采样器的名称。
        mujoco_objects (None or MujocoObject or list of MujocoObject): 单个模型或MJCF对象模型列表
        x_range (2-array of float): 指定用于均匀放置物体的(min, max)相对x范围
        y_range (2-array of float): 指定用于均匀放置物体的(min, max)相对y范围
        rotation (None or float or Iterable):
            :`None`: 添加均匀随机旋转
            :`Iterable (a,b)`: 在a和b之间均匀随机化旋转角度（弧度）
            :`value`: 添加固定角度旋转
        rotation_axis (str): 可以是'x'、'y'或'z'。应用请求旋转的轴
        ensure_object_boundary_in_range (bool):
            :`True`: 物体中心位于:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): 如果为True，将检查正确（有效）的物体放置
        reference_pos (3-array): 采样将相对于此全局(x,y,z)位置进行
        z_offset (float): 为放置添加小的z偏移。这对于不移动的固定物体很有用
            （即没有自由关节）以将它们放置在桌面上方。
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        assert len(self.x_ranges) == len(self.y_ranges)
        self.num_ranges = len(self.x_ranges)
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.idx = 0

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        为给定物体采样x位置
        参数:
            object_horizontal_radius (float): 当前正在采样的物体的半径
        返回:
            float: 采样的x位置
        """
        minimum, maximum = self.x_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        为给定物体采样y位置
        参数:
            object_horizontal_radius (float): 当前正在采样的物体的半径
        返回:
            float: 采样的y位置
        """
        minimum, maximum = self.y_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        为给定物体采样方向
        返回:
            np.array: 采样的(r,p,y)欧拉角方向
        异常:
            ValueError: [无效的旋转轴]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, collections.abc.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        if self.rotation_axis == "x":
            return np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0])
        elif self.rotation_axis == "y":
            return np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0])
        elif self.rotation_axis == "z":
            return np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])
        else:
            # 指定了无效的轴，抛出错误
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, fixtures=None, reference=None, on_top=True):
        """
        相对于此采样器的reference_pos或@reference（如果指定）进行均匀采样。
        参数:
            fixtures (dict): 当前场景中物体放置的字典，以及不应与新采样物体接触的任何其他相关
                障碍物。用于确保新生成的放置是有效的。应该是物体名称映射到(pos, quat, MujocoObject)
            reference (str or 3-tuple or None): 如果提供，采样相对放置。可以是字符串，
                对应@fixtures中找到的现有物体，或直接的(x,y,z)值。如果为None，将相对于
                此采样器的`'reference_pos'`值进行采样。
            on_top (bool): 如果为True，在参考物体顶部采样放置。这对应于当前采样的
                物体的bottom_offset + 参考物体的top_offset的z偏移
                （如果指定）
        返回:
            dict: 所有物体放置的字典，将object_names映射到(pos, quat, obj)，包括
                @fixtures中指定的放置。注意quat为(w,x,y,z)格式
        异常:
            RandomizationError: [无法放置所有物体]
            AssertionError: [参考物体名称不存在，无效输入]
        """
        # 标准化输入
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # 为分配给此采样器的所有物体采样位置和四元数
        for obj in self.mujoco_objects:
            # 首先确保当前采样的物体尚未被采样
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(horizontal_radius) + base_offset[0]
                object_y = self._sample_y(horizontal_radius) + base_offset[1]
                object_z = self.z_offset + base_offset[2]
                if on_top:
                    object_z -= bottom_offset[-1]

                # 物体不能重叠
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # 随机旋转
                    quat = self._sample_quat()

                    # 如果物体具有指定的属性，将此四元数与物体的初始旋转相乘
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # 位置有效，放下物体
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class SiteRegionRandomSampler(ObjectPositionSampler):
    """
    将所有物体放置在一个 site 上
    参数:
        name (str): 此采样器的名称。
        mujoco_objects (None or MujocoObject or list of MujocoObject): 单个模型或 MJCF 对象模型列表
        x_range (2-array of float): 指定用于均匀放置物体的相对 (min, max) x_range
        y_range (2-array of float): 指定用于均匀放置物体的相对 (min, max) y_range
        rotation (None or float or Iterable):
            :`None`: 添加均匀随机旋转
            :`Iterable (a,b)`: 在 a 和 b 之间均匀随机化旋转角度（以弧度为单位）
            :`value`: 添加固定角度旋转
        rotation_axis (str): 可以为 'x'、'y' 或 'z'。应用所请求旋转时所绕的轴
        ensure_object_boundary_in_range (bool):
            :`True`: 物体的中心位于位置：
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): 如果为 True，将检查物体放置的正确（有效）性
        reference_pos (3-array): 采样将相对于其进行的全局 (x,y,z) 位置
        z_offset (float): 为放置添加一个小的 z 偏移。这对于不移动的固定物体
            （即没有自由关节）非常有用，以将它们放置在桌面上方。
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
        sim=None,
    ):
        self.x_ranges = x_ranges
        self.y_ranges = y_ranges
        assert len(self.x_ranges) == len(self.y_ranges)
        self.num_ranges = len(self.x_ranges)
        self.rotation = rotation
        self.rotation_axis = rotation_axis
        self.idx = 0
        self.sim = sim
        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_x(self, object_horizontal_radius):
        """
        为给定物体采样x位置
        参数:
            object_horizontal_radius (float): 当前正在采样的物体的半径
        返回:
            float: 采样的x位置
        """
        minimum, maximum = self.x_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_y(self, object_horizontal_radius):
        """
        为给定物体采样y位置
        参数:
            object_horizontal_radius (float): 当前正在采样的物体的半径
        返回:
            float: 采样的y位置
        """
        minimum, maximum = self.y_ranges[self.idx]
        if self.ensure_object_boundary_in_range:
            minimum += object_horizontal_radius
            maximum -= object_horizontal_radius
        return np.random.uniform(high=maximum, low=minimum)

    def _sample_quat(self):
        """
        为给定物体采样方向
        添加多种旋转选项
        返回:
            np.array: 采样的(r,p,y)欧拉角方向
        异常:
            ValueError: [无效的旋转轴]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # 多种旋转
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # 根据请求的轴返回角度
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # 指定了无效的轴，抛出错误
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, sim, fixtures=None, reference=None, site_name="", on_top=True):
        """
        相对于此采样器的reference_pos或@reference（如果指定）进行均匀采样。
        参数:
            fixtures (dict): 当前场景中物体放置的字典，以及不应与新采样物体接触的任何其他相关
                障碍物。用于确保新生成的放置是有效的。应该是物体名称映射到(pos, quat, MujocoObject)
            reference (str or 3-tuple or None): 如果提供，采样相对放置。可以是字符串，
                对应@fixtures中找到的现有物体，或直接的(x,y,z)值。如果为None，将相对于
                此采样器的`'reference_pos'`值进行采样。
            on_top (bool): 如果为True，在参考物体顶部采样放置。这对应于当前采样的
                物体的bottom_offset + 参考物体的top_offset的z偏移
                （如果指定）
        返回:
            dict: 所有物体放置的字典，将object_names映射到(pos, quat, obj)，包括
                @fixtures中指定的放置。注意quat为(w,x,y,z)格式
        异常:
            RandomizationError: [无法放置所有物体]
            AssertionError: [参考物体名称不存在，无效输入]
        """
        # 标准化输入
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            if on_top:
                base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # 为分配给此采样器的所有物体采样位置和四元数
        for obj in self.mujoco_objects:
            # 首先确保当前采样的物体尚未被采样
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            site_x, site_y, site_z = T.quat2mat(
                T.convert_quat(ref_quat, to="xyzw")
            ) @ sim.data.get_site_xpos(site_name)
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(horizontal_radius) + base_offset[0] + site_x
                object_y = self._sample_y(horizontal_radius) + base_offset[1] + site_y
                object_z = self.z_offset + base_offset[2] + site_z
                if on_top:
                    object_z -= bottom_offset[-1]

                # 物体不能重叠
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # 随机旋转
                    quat = self._sample_quat()
                    # 如果物体具有指定的属性，将此四元数与物体的初始旋转相乘
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # 位置有效，放下物体
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class InSiteRegionRandomSampler(SiteRegionRandomSampler):
    """
    将物体放置在站点内部
    参数:
        name (str): 此采样器的名称。
        mujoco_objects (None or MujocoObject or list of MujocoObject): 单个模型或MJCF对象模型列表
        x_range (2-array of float): 指定用于均匀放置物体的(min, max)相对x范围
        y_range (2-array of float): 指定用于均匀放置物体的(min, max)相对y范围
        rotation (None or float or Iterable):
            :`None`: 添加均匀随机旋转
            :`Iterable (a,b)`: 在a和b之间均匀随机化旋转角度（弧度）
            :`value`: 添加固定角度旋转
        rotation_axis (str): 可以是'x'、'y'或'z'。应用请求旋转的轴
        ensure_object_boundary_in_range (bool):
            :`True`: 物体中心位于:
                 [uniform(min x_range + radius, max x_range - radius)], [uniform(min x_range + radius, max x_range - radius)]
            :`False`:
                [uniform(min x_range, max x_range)], [uniform(min x_range, max x_range)]
        ensure_valid_placement (bool): 如果为True，将检查正确（有效）的物体放置
        reference_pos (3-array): 采样将相对于此全局(x,y,z)位置进行
        z_offset (float): 为放置添加小的z偏移。这对于不移动的固定物体很有用
            （即没有自由关节）以将它们放置在桌面上方。
    """

    def __init__(
        self,
        name,
        mujoco_objects=None,
        x_ranges=[(0, 0)],
        y_ranges=[(0, 0)],
        rotation=None,
        rotation_axis="z",
        ensure_object_boundary_in_range=True,
        ensure_valid_placement=True,
        reference_pos=(0, 0, 0),
        z_offset=0.0,
    ):

        super().__init__(
            name=name,
            mujoco_objects=mujoco_objects,
            ensure_object_boundary_in_range=ensure_object_boundary_in_range,
            x_ranges=x_ranges,
            y_ranges=y_ranges,
            rotation=rotation,
            rotation_axis=rotation_axis,
            ensure_valid_placement=ensure_valid_placement,
            reference_pos=reference_pos,
            z_offset=z_offset,
        )

    def _sample_quat(self):
        """
        为给定物体采样方向
        添加多种旋转选项
        返回:
            np.array: 采样的(r,p,y)欧拉角方向
        异常:
            ValueError: [无效的旋转轴]
        """
        if self.rotation is None:
            rot_angle = np.random.uniform(high=2 * np.pi, low=0)
        elif isinstance(self.rotation, tuple) or isinstance(self.rotation, list):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        # 多种旋转
        elif isinstance(self.rotation, dict):
            quat = np.array(
                [0.0, 0.0, 0.0, 1.0]
            )  # \theta=0, in robosuite, quat = (x, y, z), w
            for i in range(len(self.rotation.keys())):
                rotation_axis = list(self.rotation.keys())[i]
                rot_angle = np.random.uniform(
                    high=max(self.rotation[rotation_axis]),
                    low=min(self.rotation[rotation_axis]),
                )

                if rotation_axis == "x":
                    current_quat = np.array(
                        [np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "y":
                    current_quat = np.array(
                        [0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)]
                    )
                elif rotation_axis == "z":
                    current_quat = np.array(
                        [0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)]
                    )

                quat = quat_multiply(current_quat, quat)

            return quat
        else:
            rot_angle = self.rotation

        # 根据请求的轴返回角度
        if self.rotation_axis == "x":
            return np.array([np.sin(rot_angle / 2), 0, 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "y":
            return np.array([0, np.sin(rot_angle / 2), 0, np.cos(rot_angle / 2)])
        elif self.rotation_axis == "z":
            return np.array([0, 0, np.sin(rot_angle / 2), np.cos(rot_angle / 2)])
        else:
            # 指定了无效的轴，抛出错误
            raise ValueError(
                "Invalid rotation axis specified. Must be 'x', 'y', or 'z'. Got: {}".format(
                    self.rotation_axis
                )
            )

    def sample(self, sim, fixtures=None, reference=None, site_name="", on_top=True):
        """
        相对于此采样器的reference_pos或@reference（如果指定）进行均匀采样。
        参数:
            fixtures (dict): 当前场景中物体放置的字典，以及不应与新采样物体接触的任何其他相关
                障碍物。用于确保新生成的放置是有效的。应该是物体名称映射到(pos, quat, MujocoObject)
            reference (str or 3-tuple or None): 如果提供，采样相对放置。可以是字符串，
                对应@fixtures中找到的现有物体，或直接的(x,y,z)值。如果为None，将相对于
                此采样器的`'reference_pos'`值进行采样。
            on_top (bool): 如果为True，在参考物体顶部采样放置。这对应于当前采样的
                物体的bottom_offset + 参考物体的top_offset的z偏移
                （如果指定）
        返回:
            dict: 所有物体放置的字典，将object_names映射到(pos, quat, obj)，包括
                @fixtures中指定的放置。注意quat为(w,x,y,z)格式
        异常:
            RandomizationError: [无法放置所有物体]
            AssertionError: [参考物体名称不存在，无效输入]
        """
        # 标准化输入
        placed_objects = {} if fixtures is None else copy(fixtures)
        if reference is None:
            base_offset = self.reference_pos
        elif type(reference) is str:
            assert (
                reference in placed_objects
            ), "Invalid reference received. Current options are: {}, requested: {}".format(
                placed_objects.keys(), reference
            )
            ref_pos, ref_quat, ref_obj = placed_objects[reference]
            base_offset = np.array(ref_pos)
            # if on_top:
            #     base_offset += np.array((0, 0, ref_obj.top_offset[-1]))
        else:
            base_offset = np.array(reference)
            assert (
                base_offset.shape[0] == 3
            ), "Invalid reference received. Should be (x,y,z) 3-tuple, but got: {}".format(
                base_offset
            )

        # 为分配给此采样器的所有物体采样位置和四元数
        for obj in self.mujoco_objects:
            # 首先确保当前采样的物体尚未被采样
            assert (
                obj.name not in placed_objects
            ), "Object '{}' has already been sampled!".format(obj.name)

            horizontal_radius = obj.horizontal_radius
            bottom_offset = obj.bottom_offset
            success = False
            site_x, site_y, site_z = T.quat2mat(
                T.convert_quat(ref_quat, to="xyzw")
            ) @ sim.data.get_site_xpos(site_name)
            for i in range(5000):  # 5000 retries
                self.idx = np.random.randint(self.num_ranges)
                object_x = self._sample_x(0) + base_offset[0] + site_x
                object_y = self._sample_y(0) + base_offset[1] + site_y
                object_z = self.z_offset + base_offset[2] + site_z
                if on_top:
                    object_z -= bottom_offset[-1]

                # 物体不能重叠
                location_valid = True
                if self.ensure_valid_placement:
                    for (x, y, z), _, other_obj in placed_objects.values():
                        if (
                            np.linalg.norm((object_x - x, object_y - y))
                            <= other_obj.horizontal_radius + horizontal_radius
                        ) and (
                            object_z - z <= other_obj.top_offset[-1] - bottom_offset[-1]
                        ):
                            location_valid = False
                            break

                if location_valid:
                    # 随机旋转
                    quat = self._sample_quat()

                    # 如果物体具有指定的属性，将此四元数与物体的初始旋转相乘
                    if hasattr(obj, "init_quat"):
                        quat = quat_multiply(quat, obj.init_quat)

                    # 位置有效，放下物体
                    pos = (object_x, object_y, object_z)
                    placed_objects[obj.name] = (pos, quat, obj)
                    success = True
                    break

            if not success:
                import pdb

                pdb.set_trace()
                raise RandomizationError("Cannot place all objects ):")

        return placed_objects


class SiteSequentialCompositeSampler(ObjectPositionSampler):
    """
    顺序地为每个物体采样位置。允许将多个放置初始化器链接在一起 -
    以便可以在其他物体顶部或相对于其他物体放置采样物体位置。
    参数:
        name (str): 此采样器的名称。
    """

    def __init__(self, name):
        # 采样器/参数将在稍后填充
        self.samplers = collections.OrderedDict()
        self.sample_args = collections.OrderedDict()

        super().__init__(name=name)

    def append_sampler(self, sampler, sample_args=None):
        """
        添加新的放置初始化器及其对应的@sampler和参数
        参数:
            sampler (ObjectPositionSampler): 要添加的采样器
            sample_args (None or dict): 如果指定，应该是传递给@sampler的sample()
                调用的额外参数。应将对应采样器的参数映射到值（不包括@fixtures参数）
        异常:
            AssertionError: [采样器中的物体名称]
        """
        # 验证所有添加的mujoco物体尚未被添加，并添加到此采样器的对象字典中
        for obj in sampler.mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        self.samplers[sampler.name] = sampler
        self.sample_args[sampler.name] = sample_args

    def hide(self, mujoco_objects):
        """
        从工作空间移除物体的辅助方法。
        参数:
            mujoco_objects (MujocoObject or list of MujocoObject): 要隐藏的物体
        """
        sampler = UniformRandomSampler(
            name="HideSampler",
            mujoco_objects=mujoco_objects,
            x_range=[-10, -20],
            y_range=[-10, -20],
            rotation=[0, 0],
            rotation_axis="z",
            z_offset=10,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=False,
        )
        self.append_sampler(sampler=sampler)

    def add_objects(self, mujoco_objects):
        """
        重写父类方法以确保用户不会调用此方法（所有物体应隐式属于子采样器）
        """
        raise AttributeError(
            "add_objects() should not be called for SequentialCompsiteSamplers!"
        )

    def add_objects_to_sampler(self, sampler_name, mujoco_objects):
        """
        将指定的@mujoco_objects添加到具有指定@sampler_name的子采样器。
        参数:
            sampler_name (str): 现有子采样器名称
            mujoco_objects (MujocoObject or list of MujocoObject): 要添加的物体
        """
        # 首先验证所有mujoco物体尚未被添加，并添加到此采样器的对象字典中
        mujoco_objects = (
            [mujoco_objects]
            if isinstance(mujoco_objects, MujocoObject)
            else mujoco_objects
        )
        for obj in mujoco_objects:
            assert (
                obj not in self.mujoco_objects
            ), f"Object '{obj.name}' already has sampler associated with it!"
            self.mujoco_objects.append(obj)
        # 确保sampler_name存在
        assert sampler_name in self.samplers.keys(), (
            "Invalid sub-sampler specified, valid options are: {}, "
            "requested: {}".format(self.samplers.keys(), sampler_name)
        )
        # 将mujoco物体添加到请求的子采样器
        self.samplers[sampler_name].add_objects(mujoco_objects)

    def reset(self):
        """
        重置此采样器。除了基础方法外，还遍历所有子采样器并重置它们
        """
        super().reset()
        for sampler in self.samplers.values():
            sampler.reset()

    def sample(self, sim, fixtures=None, reference=None, on_top=True):
        """
        顺序地从每个放置初始化器中采样，按它们被追加的顺序。
        参数:
            fixtures (dict): 当前场景中物体放置的字典，以及不应与新采样物体接触的任何其他相关
                障碍物。用于确保新生成的放置是有效的。应该是物体名称映射到(pos, quat, MujocoObject)
            reference (str or 3-tuple or None): 如果提供，采样相对放置。这将覆盖每个
                采样器的@reference参数（如果尚未指定）。可以是字符串，
                对应@fixtures中找到的现有物体，或直接的(x,y,z)值。如果为None，将相对于
                此采样器的`'reference_pos'`值进行采样。
            on_top (bool): 如果为True，在参考物体顶部采样放置。这将覆盖每个
                采样器的@on_top参数（如果尚未指定）。这对应于当前采样的
                物体的bottom_offset + 参考物体的top_offset的z偏移
                （如果指定）
        返回:
            dict: 所有物体放置的字典，将object_names映射到(pos, quat, obj)，包括
                @fixtures中指定的放置。注意quat为(w,x,y,z)格式
        异常:
            RandomizationError: [无法放置所有物体]
        """
        # 标准化输入
        placed_objects = {} if fixtures is None else copy(fixtures)

        # 遍历所有采样器进行采样
        for sampler, s_args in zip(self.samplers.values(), self.sample_args.values()):
            # 预处理采样器参数
            if s_args is None:
                s_args = {}
            for arg_name, arg in zip(("reference", "on_top"), (reference, on_top)):
                if arg_name not in s_args:
                    s_args[arg_name] = arg
            # 运行采样器
            new_placements = sampler.sample(sim=sim, fixtures=placed_objects, **s_args)
            # 更新放置
            placed_objects.update(new_placements)

        return placed_objects
