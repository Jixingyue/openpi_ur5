import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora


def test_lora_einsum_params_shape():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    lora0 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2))
    lora1 = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, axes=(1, 2)))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    # 确保在未使用 LoRA 时不会初始化 lora 参数。
    params = einsum.init(key, eqn, x)
    assert "lora_a" not in params["params"]
    assert "lora_b" not in params["params"]

    # 检查默认轴是否生效。
    params_lora0 = lora0.init(key, eqn, x)
    assert params_lora0["params"]["lora_a"].shape == (3, 8, 32, 2)
    assert params_lora0["params"]["lora_b"].shape == (3, 8, 2, 4)

    # 检查用户提供的轴是否生效。
    params_lora1 = lora1.init(key, eqn, x)
    assert params_lora1["params"]["lora_a"].shape == (3, 8, 2, 4)
    assert params_lora1["params"]["lora_b"].shape == (3, 2, 32, 4)


def test_lora_einsum_same_output():
    shape = (3, 8, 32, 4)  # (3KDH)
    einsum = lora.Einsum(shape)
    einsum_lora = lora.Einsum(shape, lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros))

    key = jax.random.key(0)
    x = jax.random.normal(key, (8, 64, 32))  # (BSD)
    eqn = "BSD,3KDH->3BSKH"

    params = einsum.init(key, eqn, x)
    output = einsum.apply(params, eqn, x)

    params_lora = einsum_lora.init(key, eqn, x)
    output_lora = einsum_lora.apply(params_lora, eqn, x)

    # 结果相同，因为 LoRA 参数被初始化为零。
    assert jnp.allclose(output, output_lora)


def test_lora_ffn_params_shape():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    assert params["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params["params"]["linear"].shape == (32, 8)

    params_lora = ffn_lora.init(key, x)
    assert params_lora["params"]["gating_einsum"].shape == (2, 8, 32)
    assert params_lora["params"]["linear"].shape == (32, 8)
    assert params_lora["params"]["gating_einsum_lora_a"].shape == (2, 8, 2)
    assert params_lora["params"]["gating_einsum_lora_b"].shape == (2, 2, 32)
    assert params_lora["params"]["linear_lora_a"].shape == (32, 2)
    assert params_lora["params"]["linear_lora_b"].shape == (2, 8)


def test_lora_ffn_same_output():
    ffn = lora.FeedForward(features=8, hidden_dim=32)
    ffn_lora = lora.FeedForward(
        features=8,
        hidden_dim=32,
        lora_config=lora.LoRAConfig(rank=2, init_fn=nn.initializers.zeros),
    )

    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 8))

    params = ffn.init(key, x)
    output = ffn.apply(params, x)

    params_lora = ffn_lora.init(key, x)
    output_lora = ffn_lora.apply(params_lora, x)

    assert jnp.allclose(output, output_lora)
