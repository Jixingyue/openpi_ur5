"""UR5e + D435i deployment constants."""

import numpy as np

# Control period (seconds). Match your policy / dataset control rate if possible.
DT = 1.0 / 30.0

# Resize to match openpi training pipeline (ResizeImages in model transforms).
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Default RealSense color stream (can be overridden in Args).
RS_COLOR_WIDTH = 640
RS_COLOR_HEIGHT = 480
RS_COLOR_FPS = 30

# Default moveJ target after reset (rad). **Tune for your cell** before running on hardware.
HOME_Q6 = np.asarray([-0.1, -1.2, 1.2, -1.4, -1.57, 0.0], dtype=np.float32)
