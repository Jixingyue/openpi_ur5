import h5py
import numpy as np
import json


def get_dataset_info(dataset_path, filter_key=None, verbose=True):
    # 从文件中提取演示列表
    all_filter_keys = None
    f = h5py.File(dataset_path, "r")
    if filter_key is not None:
        # 改为使用过滤键中的演示
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

    # 将演示列表按递增的回合顺序排列
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
    # 报告关于数据的统计信息
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
    if verbose:
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

        if not verbose:
            break

    f.close()
