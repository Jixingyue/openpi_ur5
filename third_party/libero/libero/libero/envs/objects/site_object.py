import os
import numpy as np

from robosuite.utils.mjcf_utils import string_to_array
import robosuite.utils.transform_utils as transform_utils

import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()


class SiteObject:
    def __init__(
        self,
        name,
        parent_name=None,
        joints=None,
        size=None,
        rgba=None,
        site_type="box",
        site_pos="0 0 0",
        site_quat="1 0 0 0",
        object_properties={},
    ):
        self.name = name
        self.parent_name = parent_name
        self.joints = joints
        self.site_pos = string_to_array(site_pos)
        self.site_quat = string_to_array(site_quat)
        self.size = size if type(size) is not str else string_to_array(size)
        self.rgba = rgba
        self.site_type = site_type
        self.object_properties = object_properties

    def in_box(self, this_position, this_mat, other_position):
        """
        检查物体是否包含在此SiteObject内。
        当CompositeObject有孔且物体应在其中一个孔内时很有用。
        通过将物体视为点，将SiteObject视为轴对齐网格来进行近似。
        参数:
            this_position: 此SiteObject的3D位置
            other_position: 要测试插入的物体的3D位置
        """

        # (TODO) Yifeng: 尺寸的变换目前有点
        # 简陋。将会深入研究。
        total_size = np.abs(this_mat @ self.size)

        ub = this_position + total_size
        lb = this_position - total_size

        lb[2] -= 0.01
        # print(np.all(other_position > lb), np.all(other_position < ub))
        # print(lb, other_position, ub)
        return np.all(other_position > lb) and np.all(other_position < ub)

    def __str__(self):
        return (
            f"Object {self.name} : \n geom type: {self.site_type} \n size: {self.size}"
        )

    def under(self, this_position, this_mat, other_position, other_height=0.10):
        """
        检查物体是否在此SiteObject上。
        当CompositeObject有孔且物体应在其中一个孔内时很有用。
        通过将物体视为点，将SiteObject视为轴对齐网格来进行近似。
        参数:
            this_position: 此SiteObject的3D位置
            other_position: 要测试插入的物体的3D位置
        """
        total_size = self.size  # np.abs(this_mat @ self.size)

        delta_position = this_mat @ (other_position - this_position)
        # print(total_size, " | ", delta_position)
        # print(total_size[2] < delta_position[2] < total_size[2] + other_height, np.all(np.abs(delta_position[:2]) < total_size[:2]))
        return total_size[2] - 0.005 < delta_position[2] < total_size[
            2
        ] + other_height and np.all(np.abs(delta_position[:2]) < total_size[:2])
