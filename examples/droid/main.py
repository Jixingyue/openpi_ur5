# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

# DROID 数据采集频率 —— 我们会降低执行速度以匹配该频率
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    # 硬件参数
    left_camera_id: str = "<your_camera_id>"  # 例如："24259877"
    right_camera_id: str = "<your_camera_id>"  # 例如："24514023"
    wrist_camera_id: str = "<your_camera_id>"  # 例如："13062452"

    # 策略参数
    external_camera: str | None = (
        None  # 向策略输入哪一部外部相机，可选值：["left", "right"]
    )

    # Rollout 参数
    max_timesteps: int = 600
    # 在一个预测的 action chunk 中执行多少个动作后才重新请求策略服务端
    # 8 通常是不错的默认值（相当于 0.5 秒的动作执行时间）。
    open_loop_horizon: int = 8

    # 远程服务端参数
    remote_host: str = "0.0.0.0"  # 指向策略服务端的 IP 地址，例如："192.168.1.100"
    remote_port: int = (
        8000  # 指向策略服务端的端口，openpi 服务端默认端口为 8000
    )


# 我们使用 Ctrl+C 可选地提前终止 rollout —— 但是，如果在策略服务端正在等待新的
# action chunk 时按下 Ctrl+C，会抛出异常并断开服务端连接。
# 该上下文管理器临时阻止 Ctrl+C，并将其延后到服务端调用完成之后再处理。
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """临时阻止键盘中断，将其延后到受保护代码执行完之后。"""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args):
    # 确保用户已指定外部相机 —— 策略仅使用一部外部相机
    assert (
        args.external_camera is not None and args.external_camera in ["left", "right"]
    ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

    # 初始化 Panda 环境。使用关节速度动作空间和夹爪位置动作空间非常重要。
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the droid env!")

    # 连接到策略服务端
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    df = pd.DataFrame(columns=["success", "duration", "video_filename"])

    while True:
        instruction = input("Enter instruction: ")

        # Rollout 参数
        actions_from_chunk_completed = 0
        pred_action_chunk = None

        # 准备保存 rollout 视频
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        video = []
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        for t_step in bar:
            start_time = time.time()
            try:
                # 获取当前观测
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # 将首个观测保存到磁盘
                    save_to_disk=t_step == 0,
                )

                video.append(curr_obs[f"{args.external_camera}_image"])

                # 如果需要预测新的 chunk，则向策略服务端发送 websocket 请求
                if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                    actions_from_chunk_completed = 0

                    # 在机器人笔记本上先对图像进行缩放，以尽量减少发送到策略服务端的数据量
                    # 并降低延迟。
                    request_data = {
                        "observation/exterior_image_1_left": image_tools.resize_with_pad(
                            curr_obs[f"{args.external_camera}_image"], 224, 224
                        ),
                        "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                        "observation/joint_position": curr_obs["joint_position"],
                        "observation/gripper_position": curr_obs["gripper_position"],
                        "prompt": instruction,
                    }

                    # 将服务端调用包裹在上下文管理器中，防止 Ctrl+C 中断它
                    # Ctrl+C 将在服务端调用完成后才被处理
                    with prevent_keyboard_interrupt():
                        # 返回一个 [10, 8] 的 action chunk：10 个关节速度动作（7 维） + 夹爪位置（1 维）
                        pred_action_chunk = policy_client.infer(request_data)["actions"]
                    assert pred_action_chunk.shape == (10, 8)

                # 从 chunk 中选择当前要执行的动作
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # 将夹爪动作二值化
                if action[-1].item() > 0.5:
                    # action[-1] = 1.0
                    action = np.concatenate([action[:-1], np.ones((1,))])
                else:
                    # action[-1] = 0.0
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                # 将 action 的所有维度限幅到 [-1, 1]
                action = np.clip(action, -1, 1)

                env.step(action)

                # 休眠以匹配 DROID 数据采集频率
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(video)
        save_filename = "video_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        success: str | float | None = None
        while not isinstance(success, float):
            success = input(
                "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
            )
            if success == "y":
                success = 1.0
            elif success == "n":
                success = 0.0

            success = float(success) / 100
            if not (0 <= success <= 1):
                print(f"Success must be a number in [0, 100] but got: {success * 100}")

        df = df.append(
            {
                "success": success,
                "duration": t_step,
                "video_filename": save_filename,
            },
            ignore_index=True,
        )

        if input("Do one more eval? (enter y or n) ").lower() != "y":
            break
        env.reset()

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    df.to_csv(csv_filename)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        # 注意：下面的 "left" 指的是双目相机对中的左侧相机。
        # 模型只在左侧双目相机上训练，因此我们仅向模型输入左侧相机图像。
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    # 去除 alpha 通道
    left_image = left_image[..., :3]
    right_image = right_image[..., :3]
    wrist_image = wrist_image[..., :3]

    # 转换为 RGB
    left_image = left_image[..., ::-1]
    right_image = right_image[..., ::-1]
    wrist_image = wrist_image[..., ::-1]

    # 除了图像观测外，同时采集本体感知状态
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    # 将图像保存到磁盘，以便在机器人运行时可以实时查看
    # 拼接成一张图以方便实时查看
    if save_to_disk:
        combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
        combined_image = Image.fromarray(combined_image)
        combined_image.save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)
