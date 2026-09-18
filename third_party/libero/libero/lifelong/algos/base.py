import os
import time

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from libero.lifelong.metric import *
from libero.lifelong.models import *
from libero.lifelong.utils import *

REGISTERED_ALGOS = {}


def register_algo(policy_class):
    """将一个策略类注册到注册表中。"""
    policy_name = policy_class.__name__.lower()
    if policy_name in REGISTERED_ALGOS:
        raise ValueError("Cannot register duplicate policy ({})".format(policy_name))

    REGISTERED_ALGOS[policy_name] = policy_class


def get_algo_class(algo_name):
    """从注册表中获取策略类。"""
    if algo_name.lower() not in REGISTERED_ALGOS:
        raise ValueError(
            "Policy class with name {} not found in registry".format(algo_name)
        )
    return REGISTERED_ALGOS[algo_name.lower()]


def get_algo_list():
    return REGISTERED_ALGOS


class AlgoMeta(type):
    """用于注册环境的元类"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # 列出所有不应在此处注册的算法。
        _unregistered_algos = []

        if cls.__name__ not in _unregistered_algos:
            register_algo(cls)
        return cls


class Sequential(nn.Module, metaclass=AlgoMeta):
    """
    序贯微调 BC 基线，也是所有终身学习算法的
    超类。
    """

    def __init__(self, n_tasks, cfg):
        super().__init__()
        self.cfg = cfg
        self.loss_scale = cfg.train.loss_scale
        self.n_tasks = n_tasks
        if not hasattr(cfg, "experiment_dir"):
            create_experiment_dir(cfg)
            print(
                f"[info] Experiment directory not specified. Creating a default one: {cfg.experiment_dir}"
            )
        self.experiment_dir = cfg.experiment_dir
        self.algo = cfg.lifelong.algo

        self.policy = get_policy_class(cfg.policy.policy_type)(cfg, cfg.shape_meta)
        self.current_task = -1

    def end_task(self, dataset, task_id, benchmark, env=None):
        """
        算法在学习每个终身任务结束时所做的事情。
        """
        pass

    def start_task(self, task):
        """
        算法在学习每个终身任务开始时所做的事情。
        """
        self.current_task = task

        # 初始化优化器和调度器
        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(), **self.cfg.train.optimizer.kwargs
        )

        self.scheduler = None
        if self.cfg.train.scheduler is not None:
            self.scheduler = eval(self.cfg.train.scheduler.name)(
                self.optimizer,
                T_max=self.cfg.train.n_epochs,
                **self.cfg.train.scheduler.kwargs,
            )

    def map_tensor_to_device(self, data):
        """将数据移动到由 self.cfg.device 指定的设备上。"""
        return TensorUtils.map_tensor(
            data, lambda x: safe_device(x, device=self.cfg.device)
        )

    def observe(self, data):
        """
        算法如何在每个数据点上进行学习。
        """
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (self.loss_scale * loss).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()

    def eval_observe(self, data):
        data = self.map_tensor_to_device(data)
        with torch.no_grad():
            loss = self.policy.compute_loss(data)
        return loss.item()

    def learn_one_task(self, dataset, task_id, benchmark, result_summary):

        self.start_task(task_id)

        # 恢复对应的操作任务 id
        gsz = self.cfg.data.task_group_size
        manip_task_ids = list(range(task_id * gsz, (task_id + 1) * gsz))

        model_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_model.pth"
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=self.cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=True,
        )

        prev_success_rate = -1.0
        best_state_dict = self.policy.state_dict()  # 目前保存最佳模型

        # 用于评估智能体在当前任务上学习的速度，这对应于
        # 新任务上成功率曲线下的面积。
        cumulated_counter = 0.0
        idx_at_best_succ = 0
        successes = []
        losses = []

        task = benchmark.get_task(task_id)
        task_emb = benchmark.get_task_emb(task_id)

        # 开始训练
        for epoch in range(0, self.cfg.train.n_epochs + 1):

            t0 = time.time()

            if epoch > 0:  # 更新
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            else:  # 仅在第 0 个 epoch 评估零样本（zero-shot）性能
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.eval_observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
            t1 = time.time()

            print(
                f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
            )

            if epoch % self.cfg.eval.eval_every == 0:  # 评估 BC 损失
                # 每隔 eval_every 个 epoch，我们在当前任务上评估智能体，
                # 然后我们选择在当前任务上表现最佳的智能体，
                # 就好像它在那个特定 epoch 之后停止学习一样。因此学习
                # 新任务的停止准则是在新任务上达到峰值性能。
                # 未来的工作可以探索如何通过同时考虑智能体在旧任务上的
                # 性能来决定这个停止 epoch。
                losses.append(training_loss)

                t0 = time.time()

                task_str = f"k{task_id}_e{epoch//self.cfg.eval.eval_every}"
                sim_states = (
                    result_summary[task_str] if self.cfg.eval.save_sim_states else None
                )
                success_rate = evaluate_one_task_success(
                    cfg=self.cfg,
                    algo=self,
                    task=task,
                    task_emb=task_emb,
                    task_id=task_id,
                    sim_states=sim_states,
                    task_str="",
                )
                successes.append(success_rate)

                if prev_success_rate < success_rate:
                    torch_save_model(self.policy, model_checkpoint_name, cfg=self.cfg)
                    prev_success_rate = success_rate
                    idx_at_best_succ = len(losses) - 1

                t1 = time.time()

                cumulated_counter += 1.0
                ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                tmp_successes = np.array(successes)
                tmp_successes[idx_at_best_succ:] = successes[idx_at_best_succ]
                print(
                    f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f} | best succ: {prev_success_rate} "
                    + f"| succ. AoC {tmp_successes.sum()/cumulated_counter:4.2f} | time: {(t1-t0)/60:4.2f}",
                    flush=True,
                )

            if self.scheduler is not None and epoch > 0:
                self.scheduler.step()

        # 加载在当前任务上表现最佳的智能体
        self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

        # 结束学习当前任务，一些算法需要后处理
        self.end_task(dataset, task_id, benchmark)

        # 返回关于前向迁移（forward transfer）的指标
        losses = np.array(losses)
        successes = np.array(successes)
        auc_checkpoint_name = os.path.join(
            self.experiment_dir, f"task{task_id}_auc.log"
        )
        torch.save(
            {
                "success": successes,
                "loss": losses,
            },
            auc_checkpoint_name,
        )

        # 假设智能体一旦达到峰值性能就停止学习
        losses[idx_at_best_succ:] = losses[idx_at_best_succ]
        successes[idx_at_best_succ:] = successes[idx_at_best_succ]
        return successes.sum() / cumulated_counter, losses.sum() / cumulated_counter

    def reset(self):
        self.policy.reset()
