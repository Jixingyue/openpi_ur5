"""Intel RealSense D435i color capture."""

from __future__ import annotations

import contextlib
import logging

import cv2
import numpy as np

from examples.ur5 import constants as _constants

logger = logging.getLogger(__name__)


class D435iColorCamera:
    """BGR8 color frames from a D435i (or compatible) via pyrealsense2."""

    def __init__(
        self,
        *,
        width: int = _constants.RS_COLOR_WIDTH,
        height: int = _constants.RS_COLOR_HEIGHT,
        fps: int = _constants.RS_COLOR_FPS,
        serial: str | None = None,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError(
                "pyrealsense2 is required for D435i capture. Install with: uv pip install pyrealsense2"
            ) from e

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(cfg)
        logger.info("RealSense pipeline started (%sx%s @ %s fps)", width, height, fps)

    def read_bgr(self) -> np.ndarray:
        """Return HxWx3 uint8 BGR (OpenCV / RealSense native)."""
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("No color frame from RealSense")
        return np.asanyarray(color.get_data())

    def read_rgb(self) -> np.ndarray:
        """Return HxWx3 uint8 RGB for the policy client."""
        bgr = self.read_bgr()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        with contextlib.suppress(AttributeError, RuntimeError):
            self._pipeline.stop()


class FakeColorCamera:
    """Offline / CI placeholder."""

    def read_rgb(self) -> np.ndarray:
        return np.zeros(
            (_constants.RS_COLOR_HEIGHT, _constants.RS_COLOR_WIDTH, 3),
            dtype=np.uint8,
        )

    def close(self) -> None:
        pass
