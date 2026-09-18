"""UR5e + D435i 部署常量。"""

import numpy as np

# 控制周期（秒）。尽可能与策略/数据集的控制频率保持一致。
DT = 1.0 / 30.0

# 缩放尺寸，与 openpi 训练管道保持一致（模型 transforms 中的 ResizeImages）。
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# RealSense 彩色流默认参数（可在 Args 中覆盖）。
RS_COLOR_WIDTH = 640
RS_COLOR_HEIGHT = 480
RS_COLOR_FPS = 30

# reset 后默认的 moveJ 目标位置（弧度）。**在硬件上运行前，请针对你的工作单元进行调优**。
HOME_Q6 = np.asarray([-0.1, -1.2, 1.2, -1.4, -1.57, 0.0], dtype=np.float32)
