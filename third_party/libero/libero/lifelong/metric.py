import copy
import gc
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.utils import *


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    准备张量观测值，作为算法的输入。
    """
    env_num = len(obs)

    data = {
        "obs": {},
        "task_emb": task_emb.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(
    cfg, algo, task, task_emb, task_id, sim_states=None, task_str=""
):
    """
    评估单个任务的成功率
    sim_states：如果不为 None，将在评估过程中跟踪所有仿真状态，
                主要用于可视化和调试目的
    task_str：   用于访问 sim_states 字典的键
    """
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # PackNet 需要预处理权重
            algo = algo.get_eval_algo(task_id)

        algo.eval()
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # 初始化评估环境
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # 尝试处理帧缓冲区问题
        env_creation = False

        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### 评估循环
        # 获取固定的初始状态以控制实验随机性
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in range(eval_loop_num):
            env.reset()
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            algo.reset()
            obs = env.set_init_state(init_states_)

            # 虚拟动作 [env_num, 7] 全为零，用于初始物理仿真
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            if task_str != "":
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                        sim_states[i * env_num + k].append(sim_state[k])

            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)

                # 记录仿真状态以用于回放目的
                if task_str != "":
                    sim_state = env.get_sim_state()
                    for k in range(env_num):
                        if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                            sim_states[i * env_num + k].append(sim_state[k])

                # 检查是否成功
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

                if all(dones):
                    break

            # 一种新形式的成功记录
            for k in range(env_num):
                if i * env_num + k < cfg.eval.n_eval:
                    num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate


def evaluate_success(cfg, algo, benchmark, task_ids, result_summary=None):
    """
    评估 task_ids 中所有任务的成功率。
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(
            cfg, algo, task_i, task_emb, i, sim_states=curr_summary, task_str=task_str
        )
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids):
    """
    评估 task_ids 中所有任务的成功率。
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


@torch.no_grad()
def evaluate_loss(cfg, algo, benchmark, datasets):
    """
    评估在所有数据集上的损失。
    """
    algo.eval()
    losses = []
    for i, dataset in enumerate(datasets):
        if cfg.lifelong.algo == "PackNet":  # PackNet 需要预处理权重
            algo = algo.get_eval_algo(task_id=i)

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            shuffle=False,
        )
        test_loss = 0
        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=cfg.device)
            )
            loss = algo.policy.compute_loss(data)
            test_loss += loss.item()
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
