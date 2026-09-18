"""
下载功能改编自 Mandlekar 等人：https://github.com/ARISE-Initiative/robomimic/blob/master/robomimic/utils/file_utils.py
"""
import os
import time
from tqdm import tqdm
from termcolor import colored
from pathlib import Path
import zipfile
import io
import urllib.request
import shutil

from libero.libero import get_libero_path

DIR = os.path.dirname(__file__)

DATASET_LINKS = {
    "libero_object": "https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip",
    "libero_goal": "https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip",
    "libero_spatial": "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip",
    "libero_100": "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip",
}


class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    检查给定的 URL 是否可达。
    来自 https://gist.github.com/dehowell/884204。
    参数:
        url (str): url 字符串
    返回:
        is_alive (bool): 如果 url 可达则为 True，否则为 False
    """
    request = urllib.request.Request(url)
    # request.get_method = lambda: 'HEAD'

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_url(url, download_dir, check_overwrite=True, is_zipfile=True):
    """
    首先检查 @url 是否可达，然后将该 url 处的文件
    下载到由 @download_dir 指定的目录中。
    使用 tqdm 在下载过程中打印进度条。
    改编自 https://github.com/tqdm/tqdm#hooks-and-callbacks 以及
    https://stackoverflow.com/a/53877507。
    参数:
        url (str): url 字符串
        download_dir (str): 文件应下载到的目录路径
        check_overwrite (bool): 如果为 True，将对下载路径进行合理性检查，以确保
            该名称的文件不存在于那里
    """

    # 检查 url 是否可达。我们需要 sleep 来确保服务器不会拒绝后续请求
    assert url_is_alive(url), "@download_url got unreachable url: {}".format(url)
    time.sleep(0.5)

    # 从 url 链接推断文件名
    fname = url.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    # 如果我们正在检查覆盖且路径已存在，
    # 我们要求用户确认他们是否想要覆盖该文件
    user_response = None
    if check_overwrite and os.path.exists(file_to_write):
        user_response = input(
            f"Warning: file {file_to_write} already exists. Overwrite? y/n\n"
        )
        # assert user_response.lower() in {"yes", "y"}, f"Did not receive confirmation. Aborting download."

    if user_response is None or user_response.lower() in {"yes", "y"}:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=fname
        ) as t:
            urllib.request.urlretrieve(
                url, filename=file_to_write, reporthook=t.update_to
            )
    if is_zipfile:
        with zipfile.ZipFile(file_to_write, "r") as archive:
            archive.extractall(path=download_dir)
        if os.path.isfile(file_to_write):
            os.remove(file_to_write)


def libero_dataset_download(datasets="all", download_dir=None, check_overwrite=True):
    """下载 libero 数据集

    参数:
        datasets (str, optional): 指定要保存哪些数据集。默认为 "all"，下载所有数据集。
        download_dir (str, optional): 存储数据集的目标位置。默认为 None，使用默认路径。
        check_overwrite (bool, optional): 检查是否覆盖数据集。默认为 True。
    """

    if download_dir is None:
        download_dir = get_libero_path("datasets")
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

        assert datasets in [
            "all",
            "libero_object",
            "libero_goal",
            "libero_spatial",
            "libero_100",
        ]

    for dataset_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_100",
    ]:
        if datasets == dataset_name or datasets == "all":
            print(f"Downloading {dataset_name}")
            download_url(
                DATASET_LINKS[dataset_name],
                download_dir=download_dir,
                check_overwrite=check_overwrite,
            )

            # (TODO)：解压文件


def check_libero_dataset(download_dir=None):
    """检查已下载数据集的完整性。

    参数:
        download_dir (str, optional): 数据集存储的路径。默认为 None，使用默认路径。

    返回:
        bool: 如果数据集成功下载则为 True，否则为 False。
    """
    if download_dir is None:
        download_dir = get_libero_path("datasets")
    check_result = True
    for dataset_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_90",
    ]:
        info_str = ""
        dataset_status = False
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            count = 0
            for path in Path(dataset_dir).glob("*.hdf5"):
                count += 1
            if (count == 10 and dataset_name != "libero_90") or (
                count == 90 and dataset_name == "libero_90"
            ):
                dataset_status = True
                info_str = colored(
                    f"[X] Dataset {dataset_name} is complete", "green", attrs=["bold"]
                )
            else:
                colored(
                    f"[?] Dataset {dataset_name} is not downloaded completely",
                    "yellow",
                    attrs=["bold"],
                )
        else:
            info_str = colored(
                f"[ ] Dataset {dataset_name} not found!!!", "red", attrs=["bold"]
            )

        print(info_str)
        check_result = check_result and dataset_status
    return check_result
