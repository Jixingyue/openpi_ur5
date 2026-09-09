"""UR5e joint streaming via RTDE (Universal Robots e-Series)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from examples.ur5 import constants as _constants

logger = logging.getLogger(__name__)


class Ur5eRtdeInterface:
    """Minimal RTDE wrapper: read 6 joint positions, stream targets with ``servoJ``."""

    def __init__(
        self,
        robot_ip: str,
        *,
        servo_time_step: float | None = None,
        dry_run: bool = False,
    ) -> None:
        self._dry_run = dry_run
        self._robot_ip = robot_ip
        self._servo_time_step = float(servo_time_step or _constants.DT)
        self._rtde_c: Any = None
        self._rtde_r: Any = None

        if dry_run:
            logger.warning("UR5e RTDE dry_run=True: no robot commands will be sent.")
            return

        try:
            import rtde_control
            import rtde_receive
        except ImportError as e:
            raise ImportError(
                "ur_rtde is required for real UR5e control. Install with: uv pip install ur-rtde"
            ) from e

        self._rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        self._rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        logger.info("Connected RTDE to UR5e at %s", robot_ip)

    def start_servo_mode(self) -> None:
        """Compatibility hook: ur_rtde streams ``servoJ`` without an explicit start call."""

    def stop_servo_mode(self) -> None:
        if self._dry_run or self._rtde_c is None:
            return
        self._rtde_c.servoStop()

    def get_actual_q(self) -> np.ndarray:
        """6 joint angles (rad)."""
        if self._dry_run or self._rtde_r is None:
            return np.zeros(6, dtype=np.float32)
        return np.asarray(self._rtde_r.getActualQ(), dtype=np.float32)

    def get_output_double_register(self, index: int) -> float:
        """UR output double register (often used by URCaps / tool I/O)."""
        if self._dry_run or self._rtde_r is None:
            return 0.0
        return float(self._rtde_r.getOutputDoubleRegister(int(index)))

    def servo_j(self, q6: np.ndarray) -> None:
        """Stream one servo setpoint (radians). Outer ``Runtime`` ``max_hz`` should match ``servo_time_step``."""
        if self._dry_run or self._rtde_c is None:
            return
        q = [float(x) for x in np.asarray(q6, dtype=np.float64).reshape(-1)[:6]]
        self._rtde_c.servoJ(q, 0.0, 0.0, self._servo_time_step, 0.1, 300)

    def move_j(self, q6: np.ndarray, *, speed: float = 0.5, acceleration: float = 0.3) -> None:
        """Blocking move (useful for reset / homing)."""
        if self._dry_run or self._rtde_c is None:
            return
        self.stop_servo_mode()
        q = [float(x) for x in np.asarray(q6, dtype=np.float64).reshape(-1)[:6]]
        self._rtde_c.moveJ(q, speed, acceleration)

    def disconnect(self) -> None:
        self.stop_servo_mode()
        if self._rtde_c is not None:
            self._rtde_c.disconnect()
            self._rtde_c = None
        if self._rtde_r is not None:
            self._rtde_r.disconnect()
            self._rtde_r = None
