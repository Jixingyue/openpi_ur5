"""
用于报告数据集信息的辅助脚本。默认情况下，会打印轨迹长度统计信息、
数据集中动作元素的最大值和最小值、存在的过滤键、环境
元数据，以及第一个演示的结构。如果传入 --verbose，它将
报告每个过滤键下的确切 demo 键，以及所有演示的结构
（而不仅仅是第一个）。

参数：
    dataset (str)：hdf5 数据集的路径

    filter_key (str)：如果提供，则对文件中与该过滤键对应的
        轨迹子集报告统计信息

    verbose (bool)：如果提供该标志，则打印更多细节，例如所有
        演示的结构（而不仅仅是第一个）

示例用法：

    # 在仓库自带的示例 hdf5 上运行脚本
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5

    # 仅在验证数据上运行脚本
    python get_dataset_info.py --dataset ../../tests/assets/test.hdf5 --filter_key valid
"""
import h5py
import json
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) if provided, report statistics on the subset of trajectories \
            in the file that correspond to this filter key",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose output",
    )
    args = parser.parse_args()

    # 从文件中提取演示列表
    filter_key = args.filter_key
    all_filter_keys = None
    f = h5py.File(args.dataset, "r")
    if filter_key is not None:
        # 改为使用来自过滤键的演示
        print("NOTE: using filter key {}".format(filter_key))
        demos = sorted(
            [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)])]
        )
    else:
        # 使用所有演示
        demos = sorted(list(f["data"].keys()))

        # 提取过滤键信息
        if "mask" in f:
            all_filter_keys = {}
            for fk in f["mask"]:
                fk_demos = sorted(
                    [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(fk)])]
                )
                all_filter_keys[fk] = fk_demos

    # 将演示列表按回合递增的顺序排列
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # 提取文件中每条轨迹的长度
    traj_lengths = []
    action_min = np.inf
    action_max = -np.inf
    for ep in demos:
        traj_lengths.append(f["data/{}/actions".format(ep)].shape[0])
        action_min = min(action_min, np.min(f["data/{}/actions".format(ep)][()]))
        action_max = max(action_max, np.max(f["data/{}/actions".format(ep)][()]))
    traj_lengths = np.array(traj_lengths)

    problem_info = json.loads(f["data"].attrs["problem_info"])

    language_instruction = "".join(problem_info["language_instruction"])
    # 报告数据的统计信息
    print("")
    print("total transitions: {}".format(np.sum(traj_lengths)))
    print("total trajectories: {}".format(traj_lengths.shape[0]))
    print("traj length mean: {}".format(np.mean(traj_lengths)))
    print("traj length std: {}".format(np.std(traj_lengths)))
    print("traj length min: {}".format(np.min(traj_lengths)))
    print("traj length max: {}".format(np.max(traj_lengths)))
    print("action min: {}".format(action_min))
    print("action max: {}".format(action_max))
    print("language instruction: {}".format(language_instruction.strip('"')))
    print("")
    print("==== Filter Keys ====")
    if all_filter_keys is not None:
        for fk in all_filter_keys:
            print("filter key {} with {} demos".format(fk, len(all_filter_keys[fk])))
    else:
        print("no filter keys")
    print("")
    if args.verbose:
        if all_filter_keys is not None:
            print("==== Filter Key Contents ====")
            for fk in all_filter_keys:
                print(
                    "filter_key {} with {} demos: {}".format(
                        fk, len(all_filter_keys[fk]), all_filter_keys[fk]
                    )
                )
        print("")
    env_meta = json.loads(f["data"].attrs["env_args"])
    print("==== Env Meta ====")
    print(json.dumps(env_meta, indent=4))
    print("")

    print("==== Dataset Structure ====")
    for ep in demos:
        print(
            "episode {} with {} transitions".format(
                ep, f["data/{}".format(ep)].attrs["num_samples"]
            )
        )
        for k in f["data/{}".format(ep)]:
            if k in ["obs", "next_obs"]:
                print("    key: {}".format(k))
                for obs_k in f["data/{}/{}".format(ep, k)]:
                    shape = f["data/{}/{}/{}".format(ep, k, obs_k)].shape
                    print(
                        "        observation key {} with shape {}".format(obs_k, shape)
                    )
            elif isinstance(f["data/{}/{}".format(ep, k)], h5py.Dataset):
                key_shape = f["data/{}/{}".format(ep, k)].shape
                print("    key: {} with shape {}".format(k, key_shape))

        if not args.verbose:
            break

    f.close()

    # 可能显示错误信息
    print("")
    if (action_min < -1.0) or (action_max > 1.0):
        raise Exception(
            "Dataset should have actions in [-1., 1.] but got bounds [{}, {}]".format(
                action_min, action_max
            )
        )
