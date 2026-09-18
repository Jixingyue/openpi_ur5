"""UR5（LeRobot）策略变换：将数据集 / 运行时的字典映射为模型格式。"""

import dataclasses
import io
from typing import Any

import einops
import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: Any) -> np.ndarray:
    """LeRobot v2 可能将图像存储为 ``uint8`` 数组、float CHW 张量，或 ``{bytes, path}`` 字典。"""
    if isinstance(image, dict) and image.get("bytes"):
        pil = Image.open(io.BytesIO(image["bytes"]))
        return np.asarray(pil.convert("RGB"), dtype=np.uint8)
    image = np.asarray(image)
    if hasattr(image, "detach"):  # torch.Tensor
        image = image.detach().cpu().numpy()
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def make_ur5_example() -> dict:
    """与 `LeRobotUR5DataConfig` repack 之后的键相匹配的随机示例（用于测试 / 调试）。"""
    return {
        "joints": np.random.rand(6).astype(np.float32),
        "gripper": np.random.rand(1).astype(np.float32),
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the cube",
    }


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """将重新打包的 UR5 字段映射为模型观测布局（参见 `examples/ur5/README.md`）。"""

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        joints = np.asarray(data["joints"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(data["gripper"], dtype=np.float32).reshape(-1)
        state = np.concatenate([joints, gripper], axis=-1)

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5Outputs(transforms.DataTransformFn):
    """将模型输出映射回 7 自由度（7-DoF）的 UR5 指令（6 个关节 + 夹爪）。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class UR5SingleCameraInputs(transforms.DataTransformFn):
    """单个第三视角相机 + 7D 本体感受（例如 LeRobot 的 ``state``），无腕部图像。

    腕部槽位会被填零；只有 ``base_0_rgb`` 被标记为有效（外加 FAST 的右侧槽位规则）。
    """

    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)
        base_image = _parse_image(data["base_rgb"])
        zero_wrist = np.zeros_like(base_image)

        is_fast = self.model_type == _model.ModelType.PI0_FAST
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": zero_wrist,
                "right_wrist_0_rgb": zero_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_ if is_fast else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


def make_ur5_single_cam_example() -> dict:
    """`LeRobotUR5SingleCamDataConfig` repack 之后的键（与本地 LeRobot v2.1 的 parquet 布局相匹配）。"""
    return {
        "state": np.random.randn(7).astype(np.float32),
        "base_rgb": np.random.randint(256, size=(64, 64, 3), dtype=np.uint8),
        "prompt": "海绵放到篮子里",
    }
