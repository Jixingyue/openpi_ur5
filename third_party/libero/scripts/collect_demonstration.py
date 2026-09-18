import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *


def collect_human_trajectory(
    env, device, arm, env_configuration, problem_info, remove_directory=[]
):
    """
    使用设备（键盘或 SpaceNav 3D 鼠标）来收集一个演示。
    rollout 轨迹以 npz 格式保存到文件中。
    修改 DataCollectionWrapper 包装器以添加新字段或更改数据格式。

    参数：
        env (MujocoEnv)：要控制的环境
        device (Device)：用于从设备接收控制信号
        arms (str)：控制哪个机械臂（例如双臂）'right' 或 'left'
        env_configuration (str)：指定的环境配置
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except:
            continue

    # ID = 2 总是对应 agentview
    env.render()

    task_completion_hold_count = (
        -1
    )  # 用于在达到目标后收集 10 个时间步的计数器
    device.start_control()

    # 循环直到我们从输入获得重置或任务完成
    saving = True
    count = 0

    while True:
        count += 1
        # 设置活动机器人
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # 获取最新的动作
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # 如果 action 为 none，那么这是一次重置，因此我们应该 break
        if action is None:
            print("Break")
            saving = False
            break

        # 运行环境 step

        env.step(action)
        env.render()
        # 如果我们完成了任务也 break
        if task_completion_hold_count == 0:
            break

        # 状态机，用于检查是否连续 10 个时间步都成功
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # 锁定状态，递减计数
            else:
                task_completion_hold_count = 10  # 在第一个成功时间步重置计数
        else:
            task_completion_hold_count = -1  # 如果没有成功则将计数器置空

    print(count)
    # 数据收集回合结束时的清理工作
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    directory, out_dir, env_info, args, remove_directory=[]
):
    """
    将保存在 @directory 中的演示汇集到
    单个 hdf5 文件中。

    hdf5 文件的结构如下。

    data (group)
        date (attribute) - 收集日期
        time (attribute) - 收集时间
        repository_version (attribute) - 收集期间使用的仓库版本
        env (attribute) - 收集演示时所处的环境名称

        demo1 (group) - 每个演示都有一个组
            model_file (attribute) - 演示的模型 xml 字符串
            states (dataset) - 展平的 mujoco 状态
            actions (dataset) - 演示期间应用的动作

        demo2 (group)
        ...

    参数：
        directory (str)：包含原始演示的目录路径。
        out_dir (str)：存储 hdf5 文件的路径。
        env_info (str)：包含环境信息的 JSON 编码字符串，
            包括控制器和机器人信息
    """

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # 将一些元数据存储在一个组的属性中
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # 将在某个时刻被填充

    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # 删除第一个动作和最后一个状态。这是因为当 DataCollector 包装器
        # 记录状态和动作时，状态是在执行该动作之后记录的。
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # 将模型 xml 存储为属性
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # 为状态和动作写入数据集
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # 写入数据集属性（元数据）
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


if __name__ == "__main__":
    # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    # 获取控制器配置
    controller_config = load_controller_config(default_controller=args.controller)

    # 创建参数配置
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # 检查我们是否在使用多臂环境，如果是则使用 env_configuration 参数

    # 创建环境
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # 用可视化包装器包装它
    env = VisualizationWrapper(env)

    # 获取控制器配置的引用并将其转换为 json 编码的字符串
    env_info = json.dumps(config)

    # 用数据收集包装器包装环境
    tmp_directory = "demonstration_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # 初始化设备
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # 创建一个新的带时间戳的目录
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{domain_name}_ln_{problem_name}_{t1}_{t2}_"
        + language_instruction.replace(" ", "_").strip('""'),
    )

    os.makedirs(new_dir)

    # 收集演示

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_human_trajectory(
            env, device, args.arm, args.config, problem_info, remove_directory
        )
        if saving:
            print(remove_directory)
            gather_demonstrations_as_hdf5(
                tmp_directory, new_dir, env_info, args, remove_directory
            )
            i += 1
