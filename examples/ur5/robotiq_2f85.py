"""基于 Modbus RTU（串口）的 Robotiq 2F-85 驱动，与 ``main_ur3_all.py`` 的用法保持一致。

寄存器布局遵循 Robotiq 2F-85 / 2F-140 Instruction Manual (Modbus RTU)：
  - 使用 FC16 从 **1000** (0x03E8) 开始写入：action + position + speed/force。
  - 使用 FC03 从 **2000** (0x07D0) 开始读取：status、echo、position/current。

依赖：``uv pip install pymodbus``（已在 ``examples/ur5/requirements.txt`` 中列出）。
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Modbus 寄存器地址（参考 Robotiq 手册）。
_REG_OUT_FIRST = 1000
_REG_IN_FIRST = 2000


def _read_regs(client: Any, address: int, count: int, *, slave: int) -> list[int]:
    # pymodbus 3.7+ 使用 ``device_id``；早期版本使用 ``slave``。
    try:
        resp = client.read_holding_registers(address, count, device_id=slave)
    except TypeError:
        resp = client.read_holding_registers(address, count, slave=slave)
    if resp.isError():
        raise RuntimeError(f"Modbus read error @ {address}: {resp}")
    return list(resp.registers)


def _write_regs(client: Any, address: int, values: list[int], *, slave: int) -> None:
    try:
        resp = client.write_registers(address, values, device_id=slave)
    except TypeError:
        resp = client.write_registers(address, values, slave=slave)
    if resp.isError():
        raise RuntimeError(f"Modbus write error @ {address} vals={values}: {resp}")


class Robotiq2F85:
    """极简 Robotiq 2F-85 驱动（接口与 ``main_ur3_all`` 中的 ``controller.robotiq_2f85`` 保持一致）。"""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 115200,
        slave_id: int = 9,
        post_command_delay_s: float = 0.006,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._slave = int(slave_id)
        self._post_cmd_delay = float(post_command_delay_s)
        self._client: Any = None
        self._activated = False

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from pymodbus.client import ModbusSerialClient
        except ImportError as e:
            raise ImportError(
                "pymodbus is required for Robotiq2F85. Install with: uv pip install pymodbus"
            ) from e

        self._client = ModbusSerialClient(
            port=self._port,
            baudrate=self._baudrate,
            parity="N",
            stopbits=1,
            bytesize=8,
            timeout=0.2,
        )
        if not self._client.connect():
            self._client = None
            raise RuntimeError(f"Failed to open Modbus RTU port {self._port!r}")
        logger.info("Robotiq Modbus RTU connected on %s @ %d baud (slave=%d)", self._port, self._baudrate, self._slave)
        return self._client

    def close_port(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
        self._activated = False

    def _sleep_cmd(self) -> None:
        if self._post_cmd_delay > 0:
            time.sleep(self._post_cmd_delay)

    def _status_byte(self, regs: list[int]) -> int:
        """首个输入寄存器的高字节（夹爪状态字节）。"""
        return (int(regs[0]) >> 8) & 0xFF

    def _gsta(self, status_byte: int) -> int:
        """激活 / 运动状态半字节（手册：状态字节中的 gSTA）。"""
        return (status_byte >> 4) & 0x03

    def _gobj(self, status_byte: int) -> int:
        """物体检测字段（手册：状态字节中的 gOBJ）。"""
        return (status_byte >> 6) & 0x03

    def get_status(self) -> dict[str, Any]:
        """返回与 ``main_ur3_all`` 日志使用的键兼容的 dict。"""
        client = self._ensure_client()
        regs = _read_regs(client, _REG_IN_FIRST, 3, slave=self._slave)
        sb = self._status_byte(regs)
        pos = (int(regs[2]) >> 8) & 0xFF
        req_echo = int(regs[1]) & 0xFF
        return {
            "position": pos,
            "requested_position": req_echo,
            "object_status": int(self._gobj(sb)),
            "gsta": int(self._gsta(sb)),
            "status_byte": int(sb),
        }

    def activate(self, *, wait: bool = True, timeout: float = 5.0) -> bool:
        """清零输出，置位 rACT，轮询直到激活完成（gSTA == 3）。"""
        client = self._ensure_client()
        _write_regs(client, _REG_OUT_FIRST, [0, 0, 0], slave=self._slave)
        self._sleep_cmd()
        _write_regs(client, _REG_OUT_FIRST, [0x0100, 0, 0], slave=self._slave)
        self._sleep_cmd()
        self._activated = False
        if not wait:
            self._activated = True
            return True
        deadline = time.time() + float(timeout)
        while time.time() < deadline:
            try:
                regs = _read_regs(client, _REG_IN_FIRST, 1, slave=self._slave)
                if self._gsta(self._status_byte(regs)) == 3:
                    self._activated = True
                    return True
            except Exception as e:
                logger.warning("Robotiq activate poll failed: %s", e)
            time.sleep(0.05)
        return False

    def move(self, *, position: int, speed: int, force: int) -> None:
        """以指定的 speed/force（0..255）移动到 ``position``（0..255）。需先成功调用 ``activate``。"""
        if not self._activated:
            raise RuntimeError("Robotiq2F85.activate() must succeed before move().")
        client = self._ensure_client()
        pos = max(0, min(255, int(position)))
        spd = max(0, min(255, int(speed)))
        frc = max(0, min(255, int(force)))
        # rACT=1, rGTO=1 → 按 Robotiq Modbus 示例，action 高字节为 0x09。
        reg0 = 0x0900
        reg1 = pos & 0xFF
        reg2 = (spd << 8) | (frc & 0xFF)
        _write_regs(client, _REG_OUT_FIRST, [reg0, reg1, reg2], slave=self._slave)
        self._sleep_cmd()
