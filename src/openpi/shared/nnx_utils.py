from collections.abc import Callable
import dataclasses
import functools
import inspect
import re
from typing import Any, ParamSpec, TypeVar

import flax.nnx as nnx
import jax

P = ParamSpec("P")
R = TypeVar("R")


def module_jit(meth: Callable[P, R], *jit_args, **jit_kwargs) -> Callable[P, R]:
    """一个高阶函数，用于 JIT 编译 `nnx.Module` 的方法，在此过程中会冻结模块的状态。

    为什么不用 `nnx.jit`？出于某些原因，直接将 `nnx.jit` 应用于 `nnx.Module` 的方法（无论绑定与否）
    会使用远比必要量更多的内存。我猜测这可能与它必须跟踪模块变更（mutations）有关。此外，
    与标准的 `jax.jit` 相比，`nnx.jit` 存在一些固有开销，因为每次调用都必须遍历 NNX 模块图。
    详见 https://github.com/google/flax/discussions/4224。

    `module_jit` 是一个通过冻结模块状态来避免上述问题的替代方案。`module_jit` 返回的函数与原始
    方法行为完全一致，只不过模块的状态被冻结在调用 `module_jit` 时的值。在 `meth` 内部对模块的
    变更仍然被允许，但这些变更会在方法调用结束后被丢弃。
    """
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")

    graphdef, state = nnx.split(meth.__self__)

    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return jitted_fn(state, *args, **kwargs)

    return wrapper


@dataclasses.dataclass(frozen=True)
class PathRegex:
    """使用正则表达式匹配路径的 NNX 过滤器。

    默认情况下，路径使用 `/` 分隔符拼接。可以通过设置 `sep` 参数来覆盖该行为。
    """

    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """将某个函数作用于状态中与过滤器匹配的叶子节点。"""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)
