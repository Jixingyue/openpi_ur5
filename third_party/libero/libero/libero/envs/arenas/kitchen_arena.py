from libero.libero.envs.arenas.style import STYLE_MAPPING
import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    string_to_array,
    xml_path_completion,
)

from libero.libero.envs.arenas.style import get_texture_filename


class KitchenTableArena(Arena):
    """
    包含一张空桌子的工作空间。


    参数:
        table_full_size (3-tuple): (L,W,H) 桌子的完整尺寸
        table_friction (3-tuple): (滑动, 扭转, 滚动) 桌子的摩擦参数
        table_offset (3-tuple): (x,y,z) 放置桌子时相对于场景中心的偏移。
            注意z值设置桌子的上限
        has_legs (bool): 桌子是否有腿
        xml (str): 加载场景的xml文件
    """

    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml="arenas/table_arena.xml",
        floor_style="light-gray",
        wall_style="light-gray-plaster",
    ):
        super().__init__(xml_path_completion(xml))

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = (
            self.bottom_pos
            + np.array([0, 0, -self.table_half_size[2]])
            + self.table_offset
        )

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.configure_location()
        texplane = self.asset.find("./texture[@name='texplane']")
        plane_file = texplane.get("file")
        plane_file = "/".join(
            plane_file.split("/")[:-1]
            + [get_texture_filename(type="floor", style=floor_style)]
        )
        texplane.set("file", plane_file)

        texwall = self.asset.find("./texture[@name='tex-wall']")
        wall_file = texwall.get("file")
        wall_file = "/".join(
            wall_file.split("/")[:-1]
            + [get_texture_filename(type="wall", style=wall_style)]
        )
        texwall.set("file", wall_file)

    def configure_location(self):
        """为此场景配置正确的位置"""
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))
        self.table_collision.set("size", array_to_string(self.table_half_size))
        self.table_collision.set("friction", array_to_string(self.table_friction))
        self.table_visual.set("size", array_to_string(self.table_half_size))
        # self.table_visual.set("rgba", array_to_string([0, 0, 0, 0]))

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

        # 如果我们不使用桌子腿，将其大小设为0
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
        else:
            # 否则，适当设置桌腿位置
            delta_x = [0.1, -0.1, -0.1, 0.1]
            delta_y = [0.1, 0.1, -0.1, -0.1]
            for leg, dx, dy in zip(self.table_legs_visual, delta_x, delta_y):
                # 如果桌子的x长度小于某个长度，将桌腿放在两端中间
                # 否则我们将其放在边缘附近
                x = 0
                if self.table_half_size[0] > abs(dx * 2.0):
                    x += np.sign(dx) * self.table_half_size[0] - dx
                # 对y重复相同的过程
                y = 0
                if self.table_half_size[1] > abs(dy * 2.0):
                    y += np.sign(dy) * self.table_half_size[1] - dy
                # 获取z值
                z = (self.table_offset[2] - self.table_half_size[2]) / 2.0
                # 设置桌腿位置
                leg.set("pos", array_to_string([x, y, -z]))
                # 设置桌腿大小
                leg.set("size", array_to_string([0.025, z]))
                # leg.set("rgba", array_to_string([0, 0, 0, 0]))

    @property
    def table_top_abs(self):
        """
        获取桌面顶部的绝对位置

        返回:
            np.array: (x,y,z) 桌子位置
        """
        return string_to_array(self.floor.get("pos")) + self.table_offset
