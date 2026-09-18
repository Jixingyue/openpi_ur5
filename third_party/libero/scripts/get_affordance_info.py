# 这是一个用于获取 xml 文件中指定的所有可供性信息的示例脚本。

import init_path
from libero.libero.envs.objects import OBJECTS_DICT
from libero.libero.utils.object_utils import get_affordance_regions

affordances = get_affordance_regions(OBJECTS_DICT)

print(affordances)
