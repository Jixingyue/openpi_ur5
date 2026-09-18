"""通过 RTDE 实现 UR5e 关节流式控制（Universal Robots e-Series）。"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from examples.ur5 import constants as _constants

logger = logging.getLogger(__name__)


class Ur5eRtdeInterface:
    """极简 RTDE 封装：读取 6 个关节位置，并使用 ``servoJ`` 流式发送目标位置。"""

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
        """兼容性钩子：ur_rtde 无需显式启动调用即可流式发送 ``servoJ``。"""

    def stop_servo_mode(self) -> None:
        if self._dry_run or self._rtde_c is None:
            return
        self._rtde_c.servoStop()

    def get_actual_q(self) -> np.ndarray:
        """6 个关节角度（弧度）。"""
        if self._dry_run or self._rtde_r is None:
            return np.zeros(6, dtype=np.float32)
        return np.asarray(self._rtde_r.getActualQ(), dtype=np.float32)

    def get_output_double_register(self, index: int) -> float:
        """UR 输出双精度寄存器（常被 URCaps / 工具 I/O 使用）。"""
        if self._dry_run or self._rtde_r is None:
            return 0.0
        return float(self._rtde_r.getOutputDoubleRegister(int(index)))

    def servo_j(self, q6: np.ndarray) -> None:
        """流式发送一个伺服设定点（弧度）。外层 ``Runtime`` 的 ``max_hz`` 应与 ``servo_time_step`` 保持一致。"""
        if self._dry_run or self._rtde_c is None:
            return
        q = [float(x) for x in np.asarray(q6, dtype=np.float64).reshape(-1)[:6]]
        self._rtde_c.servoJ(q, 0.0, 0.0, self._servo_time_step, 0.1, 300)

    def move_j(self, q6: np.ndarray, *, speed: float = 0.5, acceleration: float = 0.3) -> None:
        """阻塞式运动（适用于 reset / 回零）。"""
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
