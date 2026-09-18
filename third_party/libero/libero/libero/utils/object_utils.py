# 这是一个工具文件，包含用于检索对象信息的各种函数
from xml.etree import ElementTree

from libero.libero.envs.objects import OBJECTS_DICT, get_object_fn

EXCEPTION_DICT = {"flat_stove": "flat_stove_burner"}


def update_exception_dict(object_name, site_name):
    """更新 EXCEPTION_DICT 信息。这是为了处理可供性区域命名的一些特殊情况。

    参数:
        object_name (str): 对象名称
        site_name (str): site 名称
    """
    EXCEPTION_DICT[object_name] = site_name


def get_affordance_regions(objects, verbose=False):
    """_summary_

    参数:
        objects (MujocoObject): 一个对象字典
        verbose (bool, optional): 打印额外的调试信息。默认为 False。

    返回:
        dict: 一个包含对象名称及其可供性区域的字典。
    """
    affordances = {}
    for object_name in objects.keys():
        try:
            obj = get_object_fn(object_name)()
            # print(obj.root.findall(".//site"))
            object_affordance = []
            for site in obj.root.findall(".//site"):
                site_name = site.get("name")
                if "site" not in site_name and (
                    object_name not in EXCEPTION_DICT
                    or object_name in EXCEPTION_DICT
                    and site_name not in EXCEPTION_DICT[object_name]
                ):
                    # print(site_name)
                    # 对象初始化时对象名称已作为前缀添加。为了 bddl 文件中的一致性，将它们移除
                    object_affordance.append(site_name.replace(f"{object_name}_", ""))
            if len(object_affordance) > 0:
                affordances[object_name] = object_affordance
        except:
            if verbose:
                print(f"Skipping {object_name}")

    return affordances
