import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import json
import multiprocessing
import pprint
import time
from pathlib import Path

import hydra
import numpy as np
import wandb
import yaml
import torch
from easydict import EasyDict
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.algos import get_algo_class, get_algo_list
from libero.lifelong.models import get_policy_list
from libero.lifelong.datasets import GroupedTaskDataset, SequenceVLDataset, get_dataset
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import (
    NpEncoder,
    compute_flops,
    control_seed,
    safe_device,
    torch_load_model,
    create_experiment_dir,
    get_task_embs,
)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(hydra_cfg):
    # 预处理
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    # 将配置打印到终端
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(cfg)

    pp.pprint("Available algorithms:")
    pp.pprint(get_algo_list())

    pp.pprint("Available policies:")
    pp.pprint(get_policy_list())

    # 控制种子
    control_seed(cfg.seed)

    # 准备终身学习
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_manip_tasks = benchmark.n_tasks

    # 从基准测试准备数据集
    manip_datasets = []
    descriptions = []
    shape_meta = None

    for i in range(n_manip_tasks):
        # 目前我们假设来自同一基准测试的任务具有相同的 shape_meta
        try:
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(
                    cfg.folder, benchmark.get_task_demonstration(i)
                ),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )
        except Exception as e:
            print(
                f"[error] failed to load task {i} name {benchmark.get_task_names()[i]}"
            )
            print(f"[error] {e}")
        print(os.path.join(cfg.folder, benchmark.get_task_demonstration(i)))
        # 将语言添加到视觉数据集，因此我们称之为 vl_dataset
        task_description = benchmark.get_task(i).language
        descriptions.append(task_description)
        manip_datasets.append(task_i_dataset)

    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    gsz = cfg.data.task_group_size
    if gsz == 1:  # 每个操作任务本身就是一个终身学习任务
        datasets = [
            SequenceVLDataset(ds, emb) for (ds, emb) in zip(manip_datasets, task_embs)
        ]
        n_demos = [data.n_demos for data in datasets]
        n_sequences = [data.total_num_sequences for data in datasets]
    else:  # 将 gsz 个操作任务组合为一个终身任务，目前未使用
        assert (
            n_manip_tasks % gsz == 0
        ), f"[error] task_group_size does not divide n_tasks"
        datasets = []
        n_demos = []
        n_sequences = []
        for i in range(0, n_manip_tasks, gsz):
            dataset = GroupedTaskDataset(
                manip_datasets[i : i + gsz], task_embs[i : i + gsz]
            )
            datasets.append(dataset)
            n_demos.extend([x.n_demos for x in dataset.sequence_datasets])
            n_sequences.extend(
                [x.total_num_sequences for x in dataset.sequence_datasets]
            )

    n_tasks = n_manip_tasks // gsz  # 终身学习任务的数量
    print("\n=================== Lifelong Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {n_manip_tasks // gsz}")
    for i in range(n_tasks):
        print(f"    - Task {i+1}:")
        for j in range(gsz):
            print(f"        {benchmark.get_task(i*gsz+j).language}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    # 准备实验并更新配置
    create_experiment_dir(cfg)
    cfg.shape_meta = shape_meta

    if cfg.use_wandb:
        wandb.init(project="libero", config=cfg)
        wandb.run.name = cfg.experiment_name

    result_summary = {
        "L_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # 损失混淆矩阵
        "S_conf_mat": np.zeros((n_manip_tasks, n_manip_tasks)),  # 成功率混淆矩阵
        "L_fwd": np.zeros((n_manip_tasks,)),  # 损失 AUC，智能体学习的速度
        "S_fwd": np.zeros((n_manip_tasks,)),  # 成功率 AUC，智能体取得成功的速度
    }

    if cfg.eval.save_sim_states:
        # 用于保存评估仿真状态，以便我们稍后回放它们
        for k in range(n_manip_tasks):
            for p in range(k + 1):  # 用于在智能体学习到任务 k 时测试任务 p
                result_summary[f"k{k}_p{p}"] = [[] for _ in range(cfg.eval.n_eval)]
            for e in range(
                cfg.train.n_epochs + 1
            ):  # 用于在智能体在任务 k 上学习时，在第 e 个 epoch 测试任务 k
                if e % cfg.eval.eval_every == 0:
                    result_summary[f"k{k}_e{e//cfg.eval.eval_every}"] = [
                        [] for _ in range(cfg.eval.n_eval)
                    ]

    # 定义终身学习算法
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(n_tasks, cfg), cfg.device)
    if cfg.pretrain_model_path != "":  # 如果存在预训练模型，则加载它
        try:
            algo.policy.load_state_dict(torch_load_model(cfg.pretrain_model_path)[0])
        except:
            print(
                f"[error] cannot load pretrained model from {cfg.pretrain_model_path}"
            )
            sys.exit(0)

    print(f"[info] start lifelong learning with algo {cfg.lifelong.algo}")
    GFLOPs, MParams = compute_flops(algo, datasets[0], cfg)
    print(f"[info] policy has {GFLOPs:.1f} GFLOPs and {MParams:.1f} MParams\n")

    # 保存实验配置文件，以便我们稍后恢复或回放
    with open(os.path.join(cfg.experiment_dir, "config.json"), "w") as f:
        json.dump(cfg, f, cls=NpEncoder, indent=4)

    if cfg.lifelong.algo == "Multitask":

        algo.train()
        s_fwd, l_fwd = algo.learn_all_tasks(datasets, benchmark, result_summary)
        result_summary["L_fwd"][-1] = l_fwd
        result_summary["S_fwd"][-1] = s_fwd

        # 如果 eval.eval 为 true，则在最后在所有已见任务上进行评估
        if cfg.eval.eval:
            L = evaluate_loss(cfg, algo, benchmark, datasets)
            S = evaluate_success(
                cfg=cfg,
                algo=algo,
                benchmark=benchmark,
                task_ids=list(range(n_manip_tasks)),
                result_summary=result_summary if cfg.eval.save_sim_states else None,
            )

            result_summary["L_conf_mat"][-1] = L
            result_summary["S_conf_mat"][-1] = S

            if cfg.use_wandb:
                wandb.run.summary["success_confusion_matrix"] = result_summary[
                    "S_conf_mat"
                ]
                wandb.run.summary["loss_confusion_matrix"] = result_summary[
                    "L_conf_mat"
                ]
                wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
                wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
                wandb.run.summary.update()

            print(("[All task loss ] " + " %4.2f |" * n_tasks) % tuple(L))
            print(("[All task succ.] " + " %4.2f |" * n_tasks) % tuple(S))

            torch.save(result_summary, os.path.join(cfg.experiment_dir, f"result.pt"))
    else:
        for i in range(n_tasks):
            print(f"[info] start training on task {i}")
            algo.train()

            t0 = time.time()
            s_fwd, l_fwd = algo.learn_one_task(
                datasets[i], i, benchmark, result_summary
            )
            result_summary["S_fwd"][i] = s_fwd
            result_summary["L_fwd"][i] = l_fwd
            t1 = time.time()

            # 在学习每个任务结束时在所有已见任务上进行评估
            if cfg.eval.eval:
                L = evaluate_loss(cfg, algo, benchmark, datasets[: i + 1])
                t2 = time.time()
                S = evaluate_success(
                    cfg=cfg,
                    algo=algo,
                    benchmark=benchmark,
                    task_ids=list(range((i + 1) * gsz)),
                    result_summary=result_summary if cfg.eval.save_sim_states else None,
                )
                t3 = time.time()
                result_summary["L_conf_mat"][i][: i + 1] = L
                result_summary["S_conf_mat"][i][: i + 1] = S

                if cfg.use_wandb:
                    wandb.run.summary["success_confusion_matrix"] = result_summary[
                        "S_conf_mat"
                    ]
                    wandb.run.summary["loss_confusion_matrix"] = result_summary[
                        "L_conf_mat"
                    ]
                    wandb.run.summary["fwd_transfer_success"] = result_summary["S_fwd"]
                    wandb.run.summary["fwd_transfer_loss"] = result_summary["L_fwd"]
                    wandb.run.summary.update()

                print(
                    f"[info] train time (min) {(t1-t0)/60:.1f} "
                    + f"eval loss time {(t2-t1)/60:.1f} "
                    + f"eval success time {(t3-t2)/60:.1f}"
                )
                print(("[Task %2d loss ] " + " %4.2f |" * (i + 1)) % (i, *L))
                print(("[Task %2d succ.] " + " %4.2f |" * (i + 1)) % (i, *S))
                torch.save(
                    result_summary, os.path.join(cfg.experiment_dir, f"result.pt")
                )

    print("[info] finished learning\n")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    # 将多进程启动方法设为 'spawn'
    if multiprocessing.get_start_method(allow_none=True) != "spawn":  
        multiprocessing.set_start_method("spawn", force=True)
    main()
