"""
将数据集转换为 LeRobot 格式的极简示例脚本。

本示例使用 Libero 数据集（存储为 RLDS），但可以轻松修改为适用于您以自定义格式保存的任何其他数据。

使用方法：
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

如果您想将数据集推送到 Hugging Face Hub，可以使用以下命令：
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

注意：要运行本脚本，您需要安装 tensorflow_datasets：
`uv pip install tensorflow tensorflow_datasets`

您可以从 https://huggingface.co/datasets/openvla/modified_libero_rlds 下载原始 Libero 数据集
生成的数据集将保存到 $HF_LEROBOT_HOME 目录。
运行本转换脚本大约需要 30 分钟。
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "your_hf_username/libero"  # 输出数据集的名称，也用于 Hugging Face Hub
RAW_DATASET_NAMES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]  # 为简化起见，我们将多个 Libero 数据集合并为一个训练数据集


def main(data_dir: str, *, push_to_hub: bool = False):
    # 清理输出目录中任何已存在的数据集
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # 创建 LeRobot 数据集，定义要存储的特征
    # OpenPi 假设本体感知信息存储在 `state` 中，动作存储在 `action` 中
    # LeRobot 假设图像数据的 dtype 为 `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 遍历原始 Libero 数据集，并将 episode 写入 LeRobot 数据集
    # 您可以根据自己的数据格式修改这部分
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()

    # 可选推送到 Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
