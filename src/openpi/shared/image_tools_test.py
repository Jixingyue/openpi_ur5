import jax.numpy as jnp

from openpi.shared import image_tools


def test_resize_with_pad_shapes():
    # 测试用例 1：将图像缩放到更大的尺寸
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # 输入图像，形状为 (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # 测试用例 2：将图像缩放到更小的尺寸
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # 测试用例 3：以相同尺寸缩放图像
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)

    # 测试用例 3：使用奇数尺寸的填充缩放图像
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)
