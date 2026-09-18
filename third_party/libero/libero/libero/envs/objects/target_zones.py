import re
import numpy as np
import robosuite.utils.transform_utils as T
import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import (
    xml_path_completion,
    array_to_string,
    find_elements,
    CustomMaterial,
    add_to_dict,
    RED,
    GREEN,
    BLUE,
)

# from robosuite.models.objects import BoxObject
from libero.libero.envs.objects.site_object import SiteObject

from libero.libero.envs.base_object import (
    register_visual_change_object,
    register_object,
)


@register_object
class TargetZone(SiteObject):
    def __init__(
        self,
        name,
        zone_height=0.007,
        z_offset=0.02,
        rgba=(1, 0, 0, 1),
        joints=None,
        zone_size=(0.15, 0.05),
        zone_centroid_xy=(0, 0),
        # site_type="box",
        # site_pos="0 0 0",
        # site_quat="1 0 0 0",
    ):
        self.category_name = "_".join(
            re.sub(r"([A-Z])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.size = (zone_size[0], zone_size[1], zone_height)
        self.pos = zone_centroid_xy + (z_offset,)
        self.quat = (1, 0, 0, 0)
        super().__init__(
            name=name,
            size=self.size,
            rgba=rgba,
            site_type="box",
            site_pos=array_to_string(self.pos),
            site_quat=array_to_string(self.quat),
        )

    def in_box(self, this_position, this_mat, other_position):
        """
        检查物体是否包含在此SiteObject内。
        当CompositeObject有孔且物体应在其中一个孔内时很有用。
        通过将物体视为点，将SiteObject视为轴对齐网格来进行近似。
        参数:
            this_position: 此SiteObject的3D位置
            other_position: 要测试插入的物体的3D位置
        """

        total_size = np.abs(this_mat @ self.size)

        ub = this_position + total_size
        lb = this_position - total_size

        lb[2] -= 0.01
        return np.all(other_position > lb) and np.all(other_position < ub)

    def on_top(self, this_position, this_mat, other_position):
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
        return np.all(other_position > ub)
