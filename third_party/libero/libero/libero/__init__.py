import os
import yaml

# 这是用于定位所有基准测试相关文件的默认路径
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
)
config_file = os.path.join(libero_config_path, "config.yaml")


def get_default_path_dict(custom_location=None):
    if custom_location is None:
        benchmark_root_path = os.path.dirname(os.path.abspath(__file__))
    else:
        benchmark_root_path = custom_location

    # 这是用于定位所有默认bddl文件的默认路径
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # 这是用于定位所有默认初始状态文件的默认路径
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # 这是用于定位所有默认数据集的默认路径
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    # 这是用于定位所有默认资源的默认路径
    assets_default_path = os.path.join(benchmark_root_path, "./assets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
        "assets": assets_default_path,
    }


def get_libero_path(query_key):
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))

    # 在用户需要访问路径时给出警告
    for key in config:
        if not os.path.exists(config[key]):
            print(f"[Warning]: {key} path {config[key]} does not exist!")

    assert (
        query_key in config
    ), f"Key {query_key} not found in config file {config_file}. You need to modify it. Available keys are: {config.keys()}"
    return config[query_key]


def set_libero_default_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    print(
        f"[Warning] You are changing the default path for Libero config. This will affect all the paths in the config file."
    )
    new_config = get_default_path_dict(custom_location)
    with open(config_file, "w") as f:
        yaml.dump(new_config, f)


if not os.path.exists(libero_config_path):
    os.makedirs(libero_config_path)

if not os.path.exists(config_file):
    # 创建默认配置文件

    default_path_dict = get_default_path_dict()
    answer = input(
        "Do you want to specify a custom path for the dataset folder? (Y/N): "
    ).lower()
    if answer == "y":
        # 如果用户想指定自定义存储路径，提示他们输入
        custom_dataset_path = input(
            "Enter the path where you want to store the datasets: "
        )
        full_custom_dataset_path = os.path.join(
            os.path.abspath(os.path.expanduser(custom_dataset_path)), "datasets"
        )
        # 检查自定义存储路径是否存在，如果不存在则创建

        print("The full path of the custom storage path you entered is:")
        print(full_custom_dataset_path)
        print("Do you want to continue? (Y/N)")
        confirm_answer = input().lower()
        if confirm_answer == "y":
            if not os.path.exists(full_custom_dataset_path):
                os.makedirs(full_custom_dataset_path)
            default_path_dict["datasets"] = full_custom_dataset_path
    print("Initializing the default config file...")
    print(f"The following information is stored in the config file: {config_file}")
    # 将所有路径写入yaml文件
    with open(config_file, "w") as f:
        yaml.dump(default_path_dict, f)
    for key, value in default_path_dict.items():
        print(f"{key}: {value}")
