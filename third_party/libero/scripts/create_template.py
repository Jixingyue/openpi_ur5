"""
这是一个用于从模板创建各种文件的脚本。它旨在简化那些希望扩展 LIBERO、创建新任务的用户的流程。你仍然需要基于模板进行必要的修改以满足自己的需求，但我们的初衷是通过提供必要的模板来为你节省大量时间。
"""

import os
import xml.etree.ElementTree as ET

from libero.libero import get_libero_path
from libero.libero.envs.textures import get_texture_file_list


def create_problem_class_from_file(class_name):
    template_source_file = os.path.join(
        get_libero_path("benchmark_root"), "../../templates/problem_class_template.py"
    )
    with open(template_source_file, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        if "YOUR_CLASS_NAME" in line:
            line = line.replace("YOUR_CLASS_NAME", class_name)
        new_lines.append(line)
    with open(f"{class_name.lower()}.py", "w") as f:
        f.writelines(new_lines)
    print(f"Creating class {class_name} at the file: {class_name.lower()}.py")


def create_scene_xml_file(scene_name):
    """这只是一个帮助你快速上手的示例。对于更高级的编辑，你需要自行摸索。你可以查看所有可用的 xml 文件作为参考。"""
    template_source_file = os.path.join(
        get_libero_path("benchmark_root"), "../../templates/scene_template.xml"
    )
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(template_source_file, parser)
    root = tree.getroot()

    basic_elements = [
        ("Floor", "texplane"),
        ("Table", "tex-table"),
        ("Table legs", "tex-table-legs"),
        ("Walls", "tex-wall"),
    ]

    for (element_name, texture_name) in basic_elements:
        element = root.findall('.//texture[@name="{}"]'.format(texture_name))[0]
        type = None
        if "floor" in element_name.lower():
            type = "floor"
        elif "table" in element_name.lower():
            type = "table"
        elif "wall" in element_name.lower():
            type = "wall"
        # 如果你想更改纹理文件的路径，可以传入 texture_path 变量来更改它。
        texture_list = get_texture_file_list(type=type, texture_path="../")
        for i, (texture_name, texture_file_path) in enumerate(texture_list):
            print(f"[{i}]: {texture_name}")
        choice = int(input(f"Please select which texture to use for {element_name}: "))
        element.set("file", texture_list[choice][1])
    tree.write(f"{scene_name}.xml", encoding="utf-8")
    print(f"Creating scene {scene_name} at the file: {scene_name}.xml")
    print(
        "\n [Notice] The texture fiile paths are specified in the relative path format assuming your scene xml will be placed in the path libero/libero/assets/scenes/. "
    )
    return


def main():
    # 使用键盘选择要创建的文件
    choices = [
        "problem_class",
        "scene",
        "object",
        "arena",
    ]

    for i, choice in enumerate(choices):
        print(f"[{i}]: {choice}")
    choice = int(input("Please select which file to create: "))

    if choices[choice] == "problem_class":
        # 请用户指定类名
        class_name = input("Please specify the class name: ")
        assert " " not in class_name, "space is not allowed in the naming"
        parts = class_name.split("_")
        class_name = "_".join([part.lower().capitalize() for part in parts])
        create_problem_class_from_file(class_name)
    elif choices[choice] == "scene":
        # 请用户指定场景名称
        scene_name = input("Please specify the scene name: ")
        scene_name = scene_name.lower()
        assert " " not in scene_name, "space is not allowed in the naming"
        create_scene_xml_file(scene_name)


if __name__ == "__main__":
    main()
