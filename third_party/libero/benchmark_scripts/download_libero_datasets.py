import init_path
import argparse
import os

import libero.libero.utils.download_utils as download_utils
from libero.libero import get_libero_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        default=get_libero_path("datasets"),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100"],
        default="all",
    )
    return parser.parse_args()


def main():

    args = parse_args()

    # 请用户指定数据集的下载目录
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    # 如果不存在，则下载
    download_utils.libero_dataset_download(
        download_dir=args.download_dir, datasets=args.datasets
    )

    # (TODO) 如果数据集已存在，检查数据集是否与基准测试一致

    # 首先检查数据集是否存在
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()
