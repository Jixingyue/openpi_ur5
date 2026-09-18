import concurrent.futures
import datetime
import logging
import os
import pathlib
import re
import shutil
import stat
import subprocess
import time
import urllib.parse

import filelock
import fsspec
import fsspec.generic
import tqdm_loggable.auto as tqdm

# 用于控制缓存目录路径的环境变量，默认使用 ~/.cache/openpi。
_OPENPI_DATA_HOME = "OPENPI_DATA_HOME"
DEFAULT_CACHE_DIR = "~/.cache/openpi"

logger = logging.getLogger(__name__)


def get_cache_dir() -> pathlib.Path:
    cache_dir = pathlib.Path(os.getenv(_OPENPI_DATA_HOME, DEFAULT_CACHE_DIR)).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_folder_permission(cache_dir)
    return cache_dir


def maybe_download(url: str, *, force_download: bool = False, **kwargs) -> pathlib.Path:
    """从远程文件系统下载文件或目录到本地缓存，并返回本地路径。

    若本地文件已存在，则直接返回。

    在多个进程中并发调用此函数是安全的。
    关于缓存目录的更多细节见 `get_cache_dir`。

    Args:
        url: 待下载文件的 URL。
        force_download: 若为 True，即使文件已存在于缓存中也会重新下载。
        **kwargs: 传递给 fsspec 的额外参数。

    Returns:
        已下载文件或目录的本地路径。该路径保证存在且为绝对路径。
    """
    # 不使用 fsspec 来解析 url，以避免与远程文件系统建立不必要的连接。
    parsed = urllib.parse.urlparse(url)

    # 若这是本地路径，则短路返回。
    if parsed.scheme == "":
        path = pathlib.Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path.resolve()

    cache_dir = get_cache_dir()

    local_path = cache_dir / parsed.netloc / parsed.path.strip("/")
    local_path = local_path.resolve()

    # 检查缓存是否需要失效。
    invalidate_cache = False
    if local_path.exists():
        if force_download or _should_invalidate_cache(cache_dir, local_path):
            invalidate_cache = True
        else:
            return local_path

    try:
        lock_path = local_path.with_suffix(".lock")
        with filelock.FileLock(lock_path):
            # 确保锁文件的权限一致。
            _ensure_permissions(lock_path)
            # 首先，若现有缓存已过期，则将其移除。
            if invalidate_cache:
                logger.info(f"Removing expired cached entry: {local_path}")
                if local_path.is_dir():
                    shutil.rmtree(local_path)
                else:
                    local_path.unlink()

            if not local_path.exists():
                # 将数据下载到本地缓存。
                logger.info(f"Downloading {url} to {local_path}")
                scratch_path = local_path.with_suffix(".partial")
                # 让 openpi-assets 通过 gsutil 下载，以避免该桶的 gcsfs 认证问题。
                # 所有其他 gs:// URL（例如 big_vision）仍照常使用 gcsfs。
                if parsed.scheme == "gs" and parsed.netloc == "openpi-assets":
                    _download_gsutil(url, scratch_path, **kwargs)
                else:
                    _download_fsspec(url, scratch_path, **kwargs)

                shutil.move(scratch_path, local_path)
                _ensure_permissions(local_path)

    except PermissionError as e:
        msg = (
            f"Local file permission error was encountered while downloading {url}. "
            f"Please try again after removing the cached data using: `rm -rf {local_path}*`"
        )
        raise PermissionError(msg) from e

    return local_path


def _download_gsutil(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """若可用则使用 gsutil 从 GCS 下载文件或目录，否则回退到 gcsfs。"""
    if shutil.which("gsutil") is None:
        logger.warning(
            "gsutil not found, falling back to gcsfs. This may fail if GCP credentials are not configured correctly."
        )
        _download_fsspec(url, local_path, **kwargs)
        return
    local_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "cp", "-r", f"{url}/*", str(local_path)],
        check=True,
    )


def _download_fsspec(url: str, local_path: pathlib.Path, **kwargs) -> None:
    """从远程文件系统下载文件到本地缓存，并返回本地路径。"""
    fs, _ = fsspec.core.url_to_fs(url, **kwargs)
    info = fs.info(url)
    # 目录由以斜杠结尾的 0 字节对象表示。
    if is_dir := (info["type"] == "directory" or (info["size"] == 0 and info["name"].endswith("/"))):
        total_size = fs.du(url)
    else:
        total_size = info["size"]
    with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fs.get, url, local_path, recursive=is_dir)
        while not future.done():
            current_size = sum(f.stat().st_size for f in [*local_path.rglob("*"), local_path] if f.is_file())
            pbar.update(current_size - pbar.n)
            time.sleep(1)
        pbar.update(total_size - pbar.n)


def _set_permission(path: pathlib.Path, target_permission: int):
    """chmod 要求同时设置可执行权限，因此若权限已与目标匹配则跳过。"""
    if path.stat().st_mode & target_permission == target_permission:
        logger.debug(f"Skipping {path} because it already has correct permissions")
        return
    path.chmod(target_permission)
    logger.debug(f"Set {path} to {target_permission}")


def _set_folder_permission(folder_path: pathlib.Path) -> None:
    """将文件夹权限设置为可读、可写且可搜索。"""
    _set_permission(folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


def _ensure_permissions(path: pathlib.Path) -> None:
    """由于我们会与容器化运行时以及训练脚本共享缓存目录，因此需要确保缓存目录具有正确的权限。
    """

    def _setup_folder_permission_between_cache_dir_and_path(path: pathlib.Path) -> None:
        cache_dir = get_cache_dir()
        relative_path = path.relative_to(cache_dir)
        moving_path = cache_dir
        for part in relative_path.parts:
            _set_folder_permission(moving_path / part)
            moving_path = moving_path / part

    def _set_file_permission(file_path: pathlib.Path) -> None:
        """将所有文件设置为可读可写；若它本身是脚本，则保持其脚本属性。"""
        file_rw = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
        if file_path.stat().st_mode & 0o100:
            _set_permission(file_path, file_rw | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        else:
            _set_permission(file_path, file_rw)

    _setup_folder_permission_between_cache_dir_and_path(path)
    for root, dirs, files in os.walk(str(path)):
        root_path = pathlib.Path(root)
        for file in files:
            file_path = root_path / file
            _set_file_permission(file_path)

        for dir in dirs:
            dir_path = root_path / dir
            _set_folder_permission(dir_path)


def _get_mtime(year: int, month: int, day: int) -> float:
    """获取给定日期在 UTC 午夜时分的 mtime。"""
    date = datetime.datetime(year, month, day, tzinfo=datetime.UTC)
    return time.mktime(date.timetuple())


# 将相对路径（以正则表达式定义）映射到失效时间戳（mtime 格式）。
# 会自上而下进行部分匹配，并选择第一个匹配项。
# 缓存条目仅在其比失效时间戳更新时才会被保留。
_INVALIDATE_CACHE_DIRS: dict[re.Pattern, float] = {
    re.compile("openpi-assets/checkpoints/pi0_aloha_pen_uncap"): _get_mtime(2025, 2, 17),
    re.compile("openpi-assets/checkpoints/pi0_libero"): _get_mtime(2025, 2, 6),
    re.compile("openpi-assets/checkpoints/"): _get_mtime(2025, 2, 3),
}


def _should_invalidate_cache(cache_dir: pathlib.Path, local_path: pathlib.Path) -> bool:
    """若缓存已过期则使其失效。若缓存被失效则返回 True。"""

    assert local_path.exists(), f"File not found at {local_path}"

    relative_path = str(local_path.relative_to(cache_dir))
    for pattern, expire_time in _INVALIDATE_CACHE_DIRS.items():
        if pattern.match(relative_path):
            # 若不比失效时间戳更新，则移除。
            return local_path.stat().st_mtime <= expire_time

    return False
