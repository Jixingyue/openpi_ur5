import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_aloha_example() -> dict:
    """为 Aloha 策略创建一个随机的输入示例。"""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class AlohaInputs(transforms.DataTransformFn):
    """Aloha 策略的输入。

    预期输入：
    - images: dict[name, img]，其中 img 为 [channel, height, width]。name 必须位于 EXPECTED_CAMERAS 中。
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # 如果为 true，将会将关节和夹爪的数值从标准 Aloha 空间转换到
    # pi 内部运行时所使用的空间，基础模型就是用该空间训练的。
    adapt_to_pi: bool = True

    # 预期的相机名称。所有输入相机都必须位于此集合中。缺失的相机将会
    # 被替换为黑图，并且对应的 `image_mask` 会被设为 False。
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # 假定基础图像总是存在。
        base_image = in_images["cam_high"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # 添加额外的图像。
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": data["state"],
        }

        # 动作仅在训练期间可用。
        if "actions" in data:
            actions = np.asarray(data["actions"])
            actions = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(transforms.DataTransformFn):
    """Aloha 策略的输出。"""

    # 如果为 true，将会将关节和夹爪的数值从标准 Aloha 空间转换到
    # pi 内部运行时所使用的空间，基础模型就是用该空间训练的。
    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        # 只返回前 14 个维度。
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi)}


def _joint_flip_mask() -> np.ndarray:
    """用于在 aloha 和 pi 的关节角度之间进行转换。"""
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    # Aloha 会将夹爪位置变换到一个线性空间。下面的代码
    # 会逆转这一变换，以与在角空间上预训练的 pi0 保持一致。
    #
    # 这些常量来自 Aloha 代码：
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    # 这是 Interbotix 代码中角空间到线性空间变换的逆变换。
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # 这些常量取自 Interbotix 代码。
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # pi0 的夹爪数据在编码器计数 (2405, 3110) 之间被归一化到 (0, 1)。
    # 总共有 4096 个编码器计数，而 aloha 使用 2048 作为零点。
    # 转换为弧度意味着归一化的输入位于 (0.5476, 1.6296) 之间
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    # 将夹爪位置从 pi0 使用的格式转换为 Aloha 使用的夹爪位置。
    # 注意：单位仍然是角度，但范围不同。

    # 我们不对输出进行缩放，因为 trossen 模型的预测已经以弧度为单位。
    # 关于该常数的推导，参见 _gripper_to_angular 中的注释
    value = value + 0.5476

    # 这些常量来自 Aloha 代码：
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    # 直接对 gripper_from_angular 函数求逆。
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    # state 为 [left_arm_joint_angles, left_arm_gripper, right_arm_joint_angles, right_arm_gripper]
    # 维度大小：[6, 1, 6, 1]
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        # 如果使用浮点图像，则转换为 uint8。
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # 从 [channel, height, width] 转换为 [height, width, channel]。
        return einops.rearrange(img, "c h w -> h w c")

    images = data["images"]
    images_dict = {name: convert_image(img) for name, img in images.items()}

    data["images"] = images_dict
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # 翻转关节。
        state = _joint_flip_mask() * state
        # 逆转 Aloha 运行时所应用的夹爪变换。
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # 翻转关节。
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
