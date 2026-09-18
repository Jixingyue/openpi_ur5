import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """为 Libero 策略创建一个随机的输入示例。"""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    该类用于将模型的输入转换为预期的格式。它同时用于训练和推理。

    对于你自己的数据集，你可以拷贝该类，并根据下面的注释修改键名，从而将
    数据集中正确的元素接入模型。
    """

    # 决定将使用哪个模型。
    # 对于你自己的数据集，不要修改这一项。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 可能需要将图像解析为 uint8 (H,W,C)，因为 LeRobot 会自动
        # 将其存储为 float32 (C,H,W)，在策略推理时会被跳过。
        # 对于你自己的数据集保留这一项，但如果你的数据集将图像
        # 存储在不同于 "observation/image" 或 "observation/wrist_image" 的键中，
        # 你应该在下方修改它。
        # Pi0 模型目前支持三个图像输入：一个第三视角，
        # 以及两个腕部视角（左和右）。如果你的数据集中没有某一类
        # 图像，例如腕部图像，你可以像下方对右腕图像所做的那样，将其注释掉并替换为零。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 创建输入字典。不要修改下方字典中的键。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # 用适当形状的全零数组填充任何不存在的图像。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 我们只对 pi0 模型掩蔽填充图像，而不是 pi0-FAST。对于你自己的数据集，不要修改这一项。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 将动作填充到模型的动作维度。对于你自己的数据集保留这一项。
        # 动作仅在训练期间可用。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 将 prompt（即语言指令）传递给模型。
        # 对于你自己的数据集保留这一项（但如果指令并未存储在 "prompt" 中，请修改键名；
        # 输出字典总是需要拥有键 "prompt"）。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    该类用于将模型的输出转换回数据集特定的格式。它仅用于推理。

    对于你自己的数据集，你可以根据下面的注释修改动作维度。
    """

    def __call__(self, data: dict) -> dict:
        # 只返回前 N 个动作——由于上面我们对动作进行了填充以适配模型的
        # 动作维度，现在需要在返回字典中解析出正确数量的动作。
        # 对于 Libero，我们只返回前 7 个动作（因为其余部分是填充）。
        # 对于你自己的数据集，将 `7` 替换为你数据集的动作维度。
        return {"actions": np.asarray(data["actions"][:, :7])}
