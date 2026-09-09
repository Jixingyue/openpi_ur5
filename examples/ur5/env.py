"""UR5e + D435i environment for openpi remote inference (matches ``pi05_ur5_single_cam`` flat keys)."""

from __future__ import annotations

import logging
from typing import Any

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.ur5 import constants as _constants
from examples.ur5 import realsense_camera as _camera
from examples.ur5 import rtde_robot as _rtde_robot

logger = logging.getLogger(__name__)


class UR5RealRobotEnvironment(_environment.Environment):
    """Build observations for ``LeRobotUR5SingleCamDataConfig`` / ``UR5SingleCameraInputs``.

    The policy server expects (before server-side repack) at least:
    ``image`` (H,W,C uint8 RGB), ``state`` (7,) float32, ``prompt`` (str or bytes).

    Joint 6-vector is UR ``getActualQ``; gripper scalar is ``[0,1]`` from an optional Robotiq
    ``get_status()`` position, RTDE output register, or a constant default (see constructor args).
    """

    def __init__(
        self,
        robot: _rtde_robot.Ur5eRtdeInterface,
        camera: Any,
        *,
        prompt: str,
        home_q6: np.ndarray | None = None,
        gripper_obs_register: int = -1,
        gripper_obs_default: float = 0.85,
        render_height: int = _constants.IMAGE_HEIGHT,
        render_width: int = _constants.IMAGE_WIDTH,
        log_gripper_command_interval: int = 200,
        # Optional Robotiq 2F-85 (Modbus RTU), same role as ``main_ur3_all.py``.
        robotiq: Any | None = None,
        robotiq_speed: int = 100,
        robotiq_force: int = 25,
        robotiq_open_pos: int = 0,
        robotiq_close_pos: int = 170,
        robotiq_open_on_reset: bool = True,
    ) -> None:
        self._robot = robot
        self._camera = camera
        self._prompt = prompt
        self._home_q6 = (
            np.asarray(home_q6, dtype=np.float32).reshape(6) if home_q6 is not None else _constants.HOME_Q6.copy()
        )
        self._gripper_obs_register = gripper_obs_register
        self._gripper_obs_default = float(gripper_obs_default)
        self._render_height = render_height
        self._render_width = render_width
        self._step_idx = 0
        self._log_gripper_command_interval = log_gripper_command_interval
        self._robotiq = robotiq
        self._robotiq_speed = int(robotiq_speed)
        self._robotiq_force = int(robotiq_force)
        self._robotiq_open_pos = int(np.clip(robotiq_open_pos, 0, 255))
        self._robotiq_close_pos = int(np.clip(robotiq_close_pos, 0, 255))
        if self._robotiq_open_pos > self._robotiq_close_pos:
            self._robotiq_open_pos, self._robotiq_close_pos = self._robotiq_close_pos, self._robotiq_open_pos
        self._robotiq_open_on_reset = bool(robotiq_open_on_reset)
        self._last_robotiq_pos_cmd: int | None = None

    @override
    def reset(self) -> None:
        self._robot.stop_servo_mode()
        self._robot.move_j(self._home_q6, speed=0.25, acceleration=0.2)
        if self._robotiq is not None and self._robotiq_open_on_reset:
            try:
                self._robotiq.move(
                    position=self._robotiq_open_pos,
                    speed=self._robotiq_speed,
                    force=self._robotiq_force,
                )
                self._last_robotiq_pos_cmd = self._robotiq_open_pos
            except Exception as e:
                logger.warning("Robotiq open-on-reset failed: %s", e)
        self._robot.start_servo_mode()
        self._step_idx = 0
        logger.info("UR5 environment reset complete (moveJ home + servoStart).")

    @override
    def is_episode_complete(self) -> bool:
        return False

    def _read_gripper_obs(self) -> float:
        if self._robotiq is not None:
            try:
                st = self._robotiq.get_status()
                pos = float(st.get("position", 0))
                return float(np.clip(pos / 255.0, 0.0, 1.0))
            except Exception as e:
                logger.warning("Robotiq get_status failed, using gripper_obs_default: %s", e)
                return float(np.clip(self._gripper_obs_default, 0.0, 1.0))
        if self._gripper_obs_register >= 0:
            v = self._robot.get_output_double_register(self._gripper_obs_register)
            return float(np.clip(v, 0.0, 1.0))
        return float(np.clip(self._gripper_obs_default, 0.0, 1.0))

    @override
    def get_observation(self) -> dict:
        rgb_hwc = self._camera.read_rgb()
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_hwc, self._render_height, self._render_width)
        )
        image_chw = einops.rearrange(img, "h w c -> c h w")
        q6 = self._robot.get_actual_q()
        g = self._read_gripper_obs()
        state = np.concatenate([q6.reshape(6), np.asarray([g], dtype=np.float32)], axis=0)
        return {
            "image": np.asarray(image_chw, dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32),
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        a = np.asarray(action["actions"], dtype=np.float64).reshape(-1)
        if a.size < 7:
            raise ValueError(f"Expected 7-dim action, got shape {a.shape}")
        q_cmd = np.clip(a[:6], -6.28, 6.28)
        self._robot.servo_j(q_cmd)
        g_cmd = float(np.clip(a[6], 0.0, 1.0))
        if self._robotiq is not None:
            span = float(self._robotiq_close_pos - self._robotiq_open_pos)
            pos_cmd = round(self._robotiq_open_pos + g_cmd * span)
            pos_cmd = int(np.clip(pos_cmd, 0, 255))
            if pos_cmd != self._last_robotiq_pos_cmd:
                try:
                    self._robotiq.move(
                        position=pos_cmd,
                        speed=self._robotiq_speed,
                        force=self._robotiq_force,
                    )
                    self._last_robotiq_pos_cmd = pos_cmd
                except Exception as e:
                    logger.warning("Robotiq move failed: %s", e)
            if self._step_idx % self._log_gripper_command_interval == 0:
                logger.info("Gripper (Robotiq) normalized cmd=%.4f -> position=%d", g_cmd, pos_cmd)
        elif self._step_idx % self._log_gripper_command_interval == 0:
            logger.info(
                "Gripper command (normalized) = %.4f (no Robotiq client; set --robotiq-port or RTDE register).",
                g_cmd,
            )
        self._step_idx += 1

    def close(self) -> None:
        self._robot.stop_servo_mode()
        self._robot.disconnect()
        if self._robotiq is not None:
            try:
                self._robotiq.close_port()
            except Exception as e:
                logger.warning("Robotiq close_port failed: %s", e)
            self._robotiq = None
        if hasattr(self._camera, "close"):
            self._camera.close()


def make_camera(*, use_fake_camera: bool, serial: str | None) -> Any:
    if use_fake_camera:
        return _camera.FakeColorCamera()
    return _camera.D435iColorCamera(serial=serial)
