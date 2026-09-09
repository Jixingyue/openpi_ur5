"""UR5 (LeRobot) policy transforms: map dataset / runtime dicts to the model format."""

import dataclasses
import io
from typing import Any

import einops
import numpy as np
from PIL import Image

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image: Any) -> np.ndarray:
    """LeRobot v2 may store images as ``uint8`` arrays, float CHW tensors, or ``{bytes, path}`` dicts."""
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
    """Random example matching keys after `LeRobotUR5DataConfig` repack (for tests / debugging)."""
    return {
        "joints": np.random.rand(6).astype(np.float32),
        "gripper": np.random.rand(1).astype(np.float32),
        "base_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "wrist_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the cube",
    }


@dataclasses.dataclass(frozen=True)
class UR5Inputs(transforms.DataTransformFn):
    """Maps repacked UR5 fields to the model observation layout (see `examples/ur5/README.md`)."""

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
    """Maps model outputs back to 7-DoF UR5 commands (6 joints + gripper)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class UR5SingleCameraInputs(transforms.DataTransformFn):
    """Single third-person camera + 7D proprio (e.g. LeRobot ``state``), no wrist image.

    Wrist slots are zero-filled; only ``base_0_rgb`` is marked valid (plus FAST right-slot rule).
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
    """Keys after repack for `LeRobotUR5SingleCamDataConfig` (matches local LeRobot v2.1 parquet layout)."""
    return {
        "state": np.random.randn(7).astype(np.float32),
        "base_rgb": np.random.randint(256, size=(64, 64, 3), dtype=np.uint8),
        "prompt": "海绵放到篮子里",
    }
