"""Robotiq 2F-85 over Modbus RTU (serial), aligned with ``main_ur3_all.py`` usage.

Register layout follows Robotiq 2F-85 / 2F-140 Instruction Manual (Modbus RTU):
  - Write FC16 starting at **1000** (0x03E8): action + position + speed/force.
  - Read FC03 starting at **2000** (0x07D0): status, echo, position/current.

Requires: ``uv pip install pymodbus`` (listed in ``examples/ur5/requirements.txt``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Modbus register addresses (Robotiq manual).
_REG_OUT_FIRST = 1000
_REG_IN_FIRST = 2000


def _read_regs(client: Any, address: int, count: int, *, slave: int) -> list[int]:
    # pymodbus 3.7+ uses ``device_id``; older releases used ``slave``.
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
    """Minimal Robotiq 2F-85 driver (same surface as ``controller.robotiq_2f85`` in ``main_ur3_all``)."""

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
        """High byte of first input register (gripper status byte)."""
        return (int(regs[0]) >> 8) & 0xFF

    def _gsta(self, status_byte: int) -> int:
        """Activation / motion state nibble (manual: gSTA in status byte)."""
        return (status_byte >> 4) & 0x03

    def _gobj(self, status_byte: int) -> int:
        """Object detection field (manual: gOBJ in status byte)."""
        return (status_byte >> 6) & 0x03

    def get_status(self) -> dict[str, Any]:
        """Return dict compatible with ``main_ur3_all`` logging keys."""
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
        """Clear outputs, assert rACT, poll until activation completes (gSTA == 3)."""
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
        """Go to ``position`` (0..255) with speed/force (0..255). Requires successful ``activate``."""
        if not self._activated:
            raise RuntimeError("Robotiq2F85.activate() must succeed before move().")
        client = self._ensure_client()
        pos = max(0, min(255, int(position)))
        spd = max(0, min(255, int(speed)))
        frc = max(0, min(255, int(force)))
        # rACT=1, rGTO=1 → action high byte 0x09 per Robotiq Modbus example.
        reg0 = 0x0900
        reg1 = pos & 0xFF
        reg2 = (spd << 8) | (frc & 0xFF)
        _write_regs(client, _REG_OUT_FIRST, [reg0, reg1, reg2], slave=self._slave)
        self._sleep_cmd()
