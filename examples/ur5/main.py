"""UR5e + Intel RealSense D435i client: connect to an openpi policy server and run on hardware."""

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
    """Client for ``pi05_ur5_single_cam`` (or compatible) server: flat keys ``image``, ``state``, ``prompt``."""

    # Policy WebSocket server (GPU machine).
    host: str = "localhost"
    port: int = 8000
    api_key: str | None = None

    # UR5e controller IP (RTDE).
    robot_ip: str = "192.168.0.10"
    # If true, do not open RTDE (for testing wiring / server connectivity only).
    dry_run: bool = False

    # Must match the trained model's action chunk length (``action_horizon`` in TrainConfig).
    action_horizon: int = 10
    # Control loop rate (Hz). Should match training ``fps`` when possible.
    max_hz: float = 30.0

    num_episodes: int = 1
    max_episode_steps: int = 10_000

    # Language instruction (must match training style / language if applicable).
    prompt: str = "海绵放到篮子里"

    # RealSense: set serial to pick a specific D435i when multiple are connected.
    use_fake_camera: bool = False
    realsense_serial: str | None = None

    # Gripper observation (7th state dim): when ``robotiq_port`` is set, read from Robotiq;
    # else RTDE *output* double register, or default constant.
    gripper_obs_register: int = -1
    gripper_obs_default: float = 0.85

    # Robotiq 2F-85 (Modbus RTU). If ``robotiq_port`` is set, gripper activates at startup and
    # policy action dim 7 in ``[0,1]`` maps linearly to ``[robotiq_open_pos, robotiq_close_pos]`` (0..255).
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
