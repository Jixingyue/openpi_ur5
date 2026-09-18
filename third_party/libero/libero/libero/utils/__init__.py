import os
import yaml

# 这是用于定位所有基准测试相关文件的默认路径
libero_config_path = os.environ.get(
    "LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero")
)
config_file = os.path.join(libero_config_path, "config.yaml")


def get_path_dict(root_location=os.path.dirname(os.path.abspath(__file__))):
    benchmark_root_path = root_location

    # 这是用于定位所有默认 bddl 文件的默认路径
    bddl_files_default_path = os.path.join(benchmark_root_path, "./bddl_files")

    # 这是用于定位所有默认 bddl 文件的默认路径
    init_states_default_path = os.path.join(benchmark_root_path, "./init_files")

    # 这是用于定位所有默认数据集的默认路径
    dataset_default_path = os.path.join(benchmark_root_path, "../datasets")

    return {
        "benchmark_root": benchmark_root_path,
        "bddl_files": bddl_files_default_path,
        "init_states": init_states_default_path,
        "datasets": dataset_default_path,
    }


def get_libero_path(key):
    with open(config_file, "r") as f:
        config = dict(yaml.load(f.read(), Loader=yaml.FullLoader))
    assert key in config, f"Key {key} not found in config file {config_file}"
    return config[key]


def set_libero_path(custom_location=os.path.dirname(os.path.abspath(__file__))):
    new_config = get_path_dict(custom_location)
    with open(config_file, "w") as f:
        yaml.dump(new_config, f)


if not os.path.exists(libero_config_path):
    os.makedirs(libero_config_path)

if not os.path.exists(config_file):
    # 创建一个默认配置文件

    # 将所有路径写入一个 yaml 文件
    with open(config_file, "w") as f:
        yaml.dump(get_path_dict(), f)
