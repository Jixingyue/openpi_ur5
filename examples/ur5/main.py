"""UR5e + Intel RealSense D435i 客户端：连接到 openpi 策略服务端并在硬件上运行。"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.ur5 import env as _env
from examples.ur5 import rtde_robot as _rtde_robot


@dataclasses.dataclass
class Args:
    """``pi05_ur5_single_cam``（或兼容）服务端的客户端：扁平化 key ``image``、``state``、``prompt``。"""

    # 策略 WebSocket 服务端（GPU 机器）。
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None

    # UR5e 控制器 IP（RTDE）。
    robot_ip: str = "192.168.0.10"
    # 若为 true，则不打开 RTDE（仅用于测试线路连接 / 服务端连通性）。
    dry_run: bool = False

    # 必须与训练模型的动作块长度一致（TrainConfig 中的 ``action_horizon``）。
    action_horizon: int = 10
    # 控制循环频率（Hz）。尽可能与训练时的 ``fps`` 保持一致。
    max_hz: float = 30.0

    num_episodes: int = 1
    max_episode_steps: int = 10_000

    # 语言指令（如果适用，必须与训练风格 / 语言一致）。
    prompt: str = "海绵放到篮子里"

    # RealSense：当连接多个 D435i 时，可通过 serial 指定具体设备。
    use_fake_camera: bool = False
    realsense_serial: str | None = None

    # 夹爪观测（state 的第 7 维）：当 ``robotiq_port`` 设置时从 Robotiq 读取；
    # 否则从 RTDE *输出*双精度寄存器读取，或使用默认常量。
    gripper_obs_register: int = -1
    gripper_obs_default: float = 0.85

    # Robotiq 2F-85（Modbus RTU）。若设置了 ``robotiq_port``，启动时会激活夹爪，
    # 且策略动作的第 7 维在 ``[0,1]`` 范围内将线性映射到 ``[robotiq_open_pos, robotiq_close_pos]``（0..255）。
    robotiq_port: str | None = None
    robotiq_baudrate: int = 115200
    robotiq_slave_id: int = 9
    robotiq_speed: int = 100
    robotiq_force: int = 25
    robotiq_open_pos: int = 0
    robotiq_close_pos: int = 170
    robotiq_open_on_reset: bool = True


def main(args: Args) -> None:
    servo_dt = 1.0 / args.max_hz if args.max_hz > 0 else 1.0 / 30.0
    robot = _rtde_robot.Ur5eRtdeInterface(args.robot_ip, servo_time_step=servo_dt, dry_run=args.dry_run)
    camera = _env.make_camera(use_fake_camera=args.use_fake_camera, serial=args.realsense_serial)

    robotiq = None
    if args.robotiq_port:
        if args.dry_run:
            logging.warning("--robotiq-port is ignored while --dry-run is set.")
        else:
            from examples.ur5 import robotiq_2f85 as _robotiq_mod

            robotiq = _robotiq_mod.Robotiq2F85(
                args.robotiq_port,
                baudrate=args.robotiq_baudrate,
                slave_id=args.robotiq_slave_id,
            )
            ok = robotiq.activate(wait=True, timeout=5.0)
            if not ok:
                robotiq.close_port()
                raise RuntimeError(
                    "Robotiq2F85.activate() timed out. Check USB-RS485 wiring, slave id, and 24V power."
                )
            logging.info("Robotiq 2F-85 activated.")

    ur_env = _env.UR5RealRobotEnvironment(
        robot,
        camera,
        prompt=args.prompt,
        gripper_obs_register=args.gripper_obs_register,
        gripper_obs_default=args.gripper_obs_default,
        robotiq=robotiq,
        robotiq_speed=args.robotiq_speed,
        robotiq_force=args.robotiq_force,
        robotiq_open_pos=args.robotiq_open_pos,
        robotiq_close_pos=args.robotiq_close_pos,
        robotiq_open_on_reset=args.robotiq_open_on_reset,
    )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info("Server metadata: %s", ws_client_policy.get_server_metadata())

    metadata = ws_client_policy.get_server_metadata()
    reset_pose = metadata.get("reset_pose")
    if reset_pose is not None:
        logging.info("Server suggests reset_pose=%s (UR5 client uses constants.HOME_Q6 instead).", reset_pose)

    runtime = _runtime.Runtime(
        environment=ur_env,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        ur_env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
