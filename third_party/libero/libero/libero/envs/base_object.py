import re

OBJECTS_DICT = {}
VISUAL_CHANGE_OBJECTS_DICT = {}


def register_object(target_class):
    """我们将映射设计为不区分大小写的。"""
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    assert key not in OBJECTS_DICT
    OBJECTS_DICT[key] = target_class
    return target_class


def register_visual_change_object(target_class):
    """我们跟踪可能发生视觉变化的物体以优化代码库"""
    key = "_".join(re.sub(r"([A-Z0-9])", r" \1", target_class.__name__).split()).lower()
    VISUAL_CHANGE_OBJECTS_DICT[key] = target_class
    return target_class
