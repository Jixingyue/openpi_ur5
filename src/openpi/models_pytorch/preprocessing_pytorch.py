from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# 从 model.py 移过来的常量
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """preprocess_observation_pytorch 的 Torch.compile 兼容版本，使用简化的类型注解。

    此函数避免了可能导致 torch.compile 问题的复杂类型注解。
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: 这是一个 hack，用于同时处理 [B, C, H, W] 和 [B, H, W, C] 两种格式
        # 同时处理 [B, C, H, W] 和 [B, H, W, C] 两种格式
        is_channels_first = image.shape[1] == 3  # 检查通道是否位于维度 1

        if is_channels_first:
            # 将 [B, C, H, W] 转换为 [B, H, W, C] 以便处理
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # 将 [-1, 1] 转换为 [0, 1] 以进行 PyTorch 数据增强
            image = image / 2.0 + 0.5

            # 应用基于 PyTorch 的数据增强
            if "wrist" not in key:
                # 针对非腕相机的几何增强
                height, width = image.shape[1:3]

                # 随机裁剪并缩放
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # 随机裁剪
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # 使用张量运算而非 .item()，以兼容 torch.compile
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # 缩放回原始尺寸
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # 随机旋转（小角度）
                # 使用张量运算而非 .item()，以兼容 torch.compile
                angle = torch.rand(1, device=image.device) * 10 - 5  # -5 到 5 度之间的随机角度
                if torch.abs(angle) > 0.1:  # 仅在角度显著时才旋转
                    # 转换为弧度
                    angle_rad = angle * torch.pi / 180.0

                    # 创建旋转矩阵
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # 使用 grid_sample 应用旋转
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # 创建网格（meshgrid）
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # 扩展到批次维度
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # 应用旋转变换
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # 为 grid_sample 拼接并重塑形状
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # 针对所有相机的颜色增强
            # 随机亮度
            # 使用张量运算而非 .item()，以兼容 torch.compile
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # 0.7 到 1.3 之间的随机系数
            image = image * brightness_factor

            # 随机对比度
            # 使用张量运算而非 .item()，以兼容 torch.compile
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # 0.6 到 1.4 之间的随机系数
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # 随机饱和度（转换到 HSV，修改 S，再转换回来）
            # 为简单起见，我们只对颜色通道施加一个随机缩放
            # 使用张量运算而非 .item()，以兼容 torch.compile
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # 0.5 到 1.5 之间的随机系数
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # 将数值钳制到 [0, 1]
            image = torch.clamp(image, 0, 1)

            # 回到 [-1, 1]
            image = image * 2.0 - 1.0

        # 如果原本是通道优先格式，则转换回 [B, C, H, W] 格式
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # 获取掩码
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不进行掩蔽
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # 创建一个仅包含所需属性的简单对象，而不是使用复杂的 Observation 类
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )
