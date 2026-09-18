import collections
import numpy as np
import os
import robosuite
import xml.etree.ElementTree as ET

from copy import copy
from robosuite.utils.mjcf_utils import find_elements, xml_path_completion
from robosuite.utils.placement_samplers import ObjectPositionSampler


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
        self.idx = 0
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
        elif isinstance(self.rotation, collections.Iterable):
            rot_angle = np.random.uniform(
                high=max(self.rotation), low=min(self.rotation)
            )
        else:
            rot_angle = self.rotation

        # 根据请求的轴返回角度
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
            ref_pos, _, ref_obj = placed_objects[reference]
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
        # print(placed_objects)
        return placed_objects


def postprocess_model_xml(xml_str, cameras_dict={}, demo_generation=False):
    """
    此函数对从MuJoCo演示中收集的model.xml进行后处理，
    以确保可以找到STL文件。

    参数:
        xml_str (str): Mujoco仿真演示XML文件字符串

    返回:
        str: 后处理的xml文件字符串
    """

    path = os.path.split(robosuite.__file__)[0]
    path_split = path.split("/")

    # 替换网格和纹理文件路径
    tree = ET.fromstring(xml_str)
    root = tree
    asset = root.find("asset")
    meshes = asset.findall("mesh")
    textures = asset.findall("texture")
    all_elements = meshes + textures

    # 也替换libero的路径
    libero_path = os.getcwd() + "/libero"
    libero_path_split = libero_path.split("/")

    for elem in all_elements:
        old_path = elem.get("file")
        if old_path is None:
            continue
        old_path_split = old_path.split("/")
        if "robosuite" in old_path_split:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "robosuite"
            )  # last occurrence index
            new_path_split = path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        elif "libero" in old_path_split and demo_generation:
            ind = max(
                loc for loc, val in enumerate(old_path_split) if val == "libero"
            )  # last occurrence index
            new_path_split = libero_path_split + old_path_split[ind + 1 :]
            new_path = "/".join(new_path_split)
            elem.set("file", new_path)
        else:
            continue

    # cameras = root.find("worldbody").findall("camera")
    cameras = find_elements(root=tree, tags="camera", return_first=False)
    for camera in cameras:
        camera_name = camera.get("name")
        if camera_name in cameras_dict:
            camera.set("name", camera_name)
            camera.set("pos", cameras_dict[camera_name]["pos"])
            camera.set("quat", cameras_dict[camera_name]["quat"])
            camera.set("mode", "fixed")

    return ET.tostring(root, encoding="utf8").decode("utf8")


def rectangle2xyrange(rect_ranges):
    x_ranges = []
    y_ranges = []
    for rect_range in rect_ranges:
        x_ranges.append([rect_range[0], rect_range[2]])
        y_ranges.append([rect_range[1], rect_range[3]])
    return x_ranges, y_ranges
