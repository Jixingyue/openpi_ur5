from .base_region_sampler import *
from .workspace_region_sampler import *
from .object_property_sampler import *

"""

为不同的问题域定义不同的区域。

注册区域采样器的命名规范：
key: 小写命名，每个单词用连字符分隔
value: 小写命名，{problem_name}.{region_sampler_class_name}

"""
REGION_SAMPLERS = {
    "libero_tabletop_manipulation": {"table": TableRegionSampler},
    "libero_floor_manipulation": {"floor": TableRegionSampler},
    "libero_coffee_table_manipulation": {"coffee_table": TableRegionSampler},
    "libero_living_room_tabletop_manipulation": {
        "living_room_table": Libero100TableRegionSampler
    },
    "libero_study_tabletop_manipulation": {"study_table": Libero100TableRegionSampler},
    "libero_kitchen_tabletop_manipulation": {
        "kitchen_table": Libero100TableRegionSampler
    },
}


def update_region_samplers(
    problem_name, region_sampler_name, region_sampler_class_name
):
    """
    这是用于注册自定义区域采样器的，无需添加/修改原始代码库。
    """
    if problem_name not in REGION_SAMPLERS:
        REGION_SAMPLERS[problem_name] = {}
    REGION_SAMPLERS[problem_name][region_sampler_name] = eval(
        f"{problem_name}.{region_sampler_class_name}"
    )


def get_region_samplers(problem_name, region_sampler_name):
    return REGION_SAMPLERS[problem_name][region_sampler_name]
