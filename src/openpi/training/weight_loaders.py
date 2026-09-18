import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """加载模型权重。

        Args:
            params: 模型的参数。这是一个由 array-like 对象组成的嵌套结构，用于
                表示模型的参数。

        Returns:
            加载后的参数。其结构必须与 `params` 完全相同。如果只返回参数的一个子集，
            加载器必须将加载的参数与 `params` 合并。
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """从 checkpoint 加载一整套权重。

    兼容：
      训练得到的 checkpoint：
        例："./checkpoints/<config>/<exp>/<step>/params"
      发布的 checkpoint：
        例："gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # 我们加载的是 np.ndarray，并依赖训练代码正确地转换和分片（shard）这些参数。
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # 添加所有缺失的 LoRA 权重。
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """从官方 PaliGemma checkpoint 加载权重。

    这会以名称相似的权重覆写现有的权重，同时保留所有额外的权重不变。
    这让我们能够支持 Pi0 模型所使用的动作专家（action expert）。
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # 添加所有缺失的权重。
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """将加载的参数与参考参数合并。

    Args:
        loaded_params: 待合并的参数。
        params: 参考参数。
        missing_regex: 一个正则表达式，匹配所有应从参考参数合并过来的缺失键。

    Returns:
        一个包含合并后参数的新字典。
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # 首先，取出所有属于参考权重子集的权重。
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # 然后，根据 missing regex 合并任何缺失的权重。
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
