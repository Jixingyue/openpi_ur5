import contextlib
import functools as ft
import inspect
from typing import TypeAlias, TypeVar, cast

import beartype
import jax
import jax._src.tree_util as private_tree_util
import jax.core
from jaxtyping import ArrayLike
from jaxtyping import Bool  # noqa: F401
from jaxtyping import DTypeLike  # noqa: F401
from jaxtyping import Float
from jaxtyping import Int  # noqa: F401
from jaxtyping import Key  # noqa: F401
from jaxtyping import Num  # noqa: F401
from jaxtyping import PyTree
from jaxtyping import Real  # noqa: F401
from jaxtyping import UInt8  # noqa: F401
from jaxtyping import config
from jaxtyping import jaxtyped
import jaxtyping._decorator
import torch

# 修补 jaxtyping 以处理 https://github.com/patrick-kidger/jaxtyping/issues/277。
# 问题在于，由于 JAX 的 tracing 操作，自定义的 PyTree 节点有时会被初始化为任意类型（例如
# `jax.ShapeDtypeStruct`、`jax.Sharding`，甚至 <object>）。该补丁会在堆栈跟踪中包含
# `jax._src.tree_util` 时跳过类型检查，而这种情况只应出现在树还原（unflattening）期间。
_original_check_dataclass_annotations = jaxtyping._decorator._check_dataclass_annotations  # noqa: SLF001
# 重新定义 Array，使其同时包含 JAX 数组和 PyTorch 张量
Array = jax.Array | torch.Tensor


def _check_dataclass_annotations(self, typechecker):
    if not any(
        frame.frame.f_globals.get("__name__") in {"jax._src.tree_util", "flax.nnx.transforms.compilation"}
        for frame in inspect.stack()
    ):
        return _original_check_dataclass_annotations(self, typechecker)
    return None


jaxtyping._decorator._check_dataclass_annotations = _check_dataclass_annotations  # noqa: SLF001

KeyArrayLike: TypeAlias = jax.typing.ArrayLike
Params: TypeAlias = PyTree[Float[ArrayLike, "..."]]

T = TypeVar("T")


# 运行时类型检查装饰器
def typecheck(t: T) -> T:
    return cast(T, ft.partial(jaxtyped, typechecker=beartype.beartype)(t))


@contextlib.contextmanager
def disable_typechecking():
    initial = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)  # noqa: FBT003
    yield
    config.update("jaxtyping_disable", initial)


def check_pytree_equality(*, expected: PyTree, got: PyTree, check_shapes: bool = False, check_dtypes: bool = False):
    """检查两个 PyTree 是否具有相同的结构，并可选择检查形状和 dtype。相比直接对结构不同的
    PyTree 使用 `jax.tree.map`，该函数能生成更友好的错误信息。
    """

    if errors := list(private_tree_util.equality_errors(expected, got)):
        raise ValueError(
            "PyTrees have different structure:\n"
            + (
                "\n".join(
                    f"   - at keypath '{jax.tree_util.keystr(path)}': expected {thing1}, got {thing2}, so {explanation}.\n"
                    for path, thing1, thing2, explanation in errors
                )
            )
        )

    if check_shapes or check_dtypes:

        def check(kp, x, y):
            if check_shapes and x.shape != y.shape:
                raise ValueError(f"Shape mismatch at {jax.tree_util.keystr(kp)}: expected {x.shape}, got {y.shape}")

            if check_dtypes and x.dtype != y.dtype:
                raise ValueError(f"Dtype mismatch at {jax.tree_util.keystr(kp)}: expected {x.dtype}, got {y.dtype}")

        jax.tree_util.tree_map_with_path(check, expected, got)
