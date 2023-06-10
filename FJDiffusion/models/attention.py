import einops
import jax.nn.initializers
from flax import linen as nn
from jax import numpy as jnp
from typing import List, Tuple, Any
from functools import partial
from einops import rearrange
from fjutils.utils import with_sharding_constraint
from jax.experimental.pjit import PartitionSpec as PS


class SelfAttentionWC(nn.Module):
    features: int
    num_attention_heads: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None
    residual_f: int = -1
    in_proj_b: bool = False
    out_proj_b: bool = True

    def setup(self) -> None:
        self.qkv = nn.Dense(self.features * 3, use_bias=self.in_proj_b)
        self.o = nn.Dense(self.features, use_bias=self.out_proj_b)
        self.head_dim = self.features // self.num_attention_heads

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_attention_heads, self.head_dim))

    def __call__(self, x, use_causal_mask: bool = False):
        shape = x.shape
        b, s, d = x.shape

        q, k, v = jnp.split(self.qkv(x), 3, axis=-1)
        q = with_sharding_constraint(q, PS(('dp', 'fsdp'), None, 'mp'))
        v = with_sharding_constraint(v, PS(('dp', 'fsdp'), None, 'mp'))
        k = with_sharding_constraint(k, PS(('dp', 'fsdp'), None, 'mp'))
        q = einops.rearrange(q, 'b s (h d) -> b s h d', h=self.num_attention_heads)
        k = einops.rearrange(k, 'b s (h d) -> b s h d', h=self.num_attention_heads)
        v = einops.rearrange(v, 'b s (h d) -> b s h d', h=self.num_attention_heads)

        mask = jnp.where(nn.attention.make_causal_mask(jnp.ones((b, s))) == 1, 0,
                         jnp.finfo(q).min) if use_causal_mask else None

        attn = nn.attention.dot_product_attention_weights(
            query=q, key=k,
            precision=self.precision,
            dtype=self.dtype,
            mask=mask
        )
        attn = with_sharding_constraint(attn, PS(('dp', 'fsdp'), 'mp', None, None))
        attn = jnp.einsum('...hqk,...khd->...qhd', attn, v).reshape(shape)
        return self.o(attn)


class SelfAttention(nn.Module):
    features: int
    num_attention_heads: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None
    residual_f: int = -1

    def setup(self):
        self.head_dim = self.features // self.num_attention_heads

        self.q_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )
        self.k_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )
        self.v_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )
        self.o_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.features,))

    def __call__(
            self,
            hidden_states,

            deterministic: bool = False,

    ):
        shape = hidden_states.shape
        if hidden_states.ndim != 3:
            b = hidden_states.shape[0]
            hidden_states = hidden_states.reshape(b, -1, self.num_attention_heads * self.head_dim)
        xq, xk, xv = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))
        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)
        attn_weights = nn.attention.dot_product_attention_weights(
            xq,
            xk,
            deterministic=deterministic,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            precision=self.precision,
        )
        attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)
        attn_output = self.o_proj(self._merge_heads(attn_output)).reshape(shape)

        return attn_output


class CrossAttention(nn.Module):
    features: int
    num_attention_heads: int
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision = None
    residual_f: int = -1

    def setup(self):
        self.head_dim = self.features // self.num_attention_heads

        self.q_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )
        self.k_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )
        self.v_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )
        self.o_proj = nn.Dense(
            self.num_attention_heads * self.head_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(),
            precision=self.precision,
        )

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.num_attention_heads, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.features,))

    def _check_input(self, x):
        if x.ndim != 3:
            b = x.shape[0]
            x = x.reshape(b, -1, self.num_attention_heads * self.head_dim)
        return x

    def __call__(
            self,
            x,
            y,
            deterministic: bool = False,

    ):
        shape = x.shape
        y = self._check_input(y)
        x = self._check_input(x)
        xq, xk, xv = self.q_proj(x), self.k_proj(y), self.v_proj(y)
        xq = with_sharding_constraint(xq, PS(("dp", "fsdp"), None, "mp"))
        xk = with_sharding_constraint(xk, PS(("dp", "fsdp"), None, "mp"))
        xv = with_sharding_constraint(xv, PS(("dp", "fsdp"), None, "mp"))
        xq = self._split_heads(xq)
        xk = self._split_heads(xk)
        xv = self._split_heads(xv)
        attn_weights = nn.attention.dot_product_attention_weights(
            xq,
            xk,
            deterministic=deterministic,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            precision=self.precision,
        )
        attn_weights = with_sharding_constraint(attn_weights, PS(("dp", "fsdp"), "mp", None, None))
        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, xv, precision=self.precision)
        attn_output = self.o_proj(self._merge_heads(attn_output)).reshape(shape)

        return attn_output


class QKVAttention(nn.Module):
    num_attention_heads: int

    def __call__(self, qkv: jnp.DeviceArray):
        batch, width, length = qkv.shape
        assert width % (3 * self.num_attention_heads) == 0
        ch = width // (3 * self.num_attention_heads)
        q, k, v = jnp.split(qkv, axis=1)
        scale = jax.lax.rsqrt(jnp.sqrt(ch))
        weight = jnp.einsum(
            'bct,bcg-> bgt',
            (q * scale).reshape(batch * self.num_attention_heads, ch, length),
            (k * scale).reshape(batch * self.num_attention_heads, ch, length)
        )
        weight = jax.nn.softmax(weight.astype(jnp.float32), -1).astype(weight.dtype)
        attention = jnp.einsum(
            'bct,bcs->bts',
            weight,
            v.reshape(batch * self.num_attention_heads, ch, length)
        )
        return attention.reshape(batch, -1, length)


class AttentionBlockQKV(nn.Module):
    num_attention_heads: int
    features: int

    def setup(self) -> None:
        self.norm = nn.LayerNorm()
        self.qkv = nn.Conv(self.features * 3, kernel_size=(1,))
        self.attn = QKVAttention(num_attention_heads=self.num_attention_heads)
        self.out = nn.Conv(self.features, kernel_size=(1,))

    def __call__(self, x):
        b, t, *c = x.shape
        x = x.reshape(b, t, -1)
        qkv = self.qkv(self.norm(x.astype(jnp.float32)).astype(x.dtype))
        qkv = self.attn(qkv=qkv)
        return (self.out(qkv) + x).reshape(b, t, *c)


class ResidualBlock(nn.Module):
    features: int
    idn: bool = False

    def setup(self) -> None:
        self.group_norm_1 = nn.GroupNorm()
        self.conv_1 = nn.Conv(features=self.features, kernel_size=(3, 3), padding=(1, 1))
        self.time = nn.Dense(self.features)
        self.group_norm_2 = nn.GroupNorm()
        self.conv_2 = nn.Conv(features=self.features, kernel_size=(3, 3), padding=(1, 1))
        if self.idn:
            self.out = nn.Conv(features=self.features, kernel_size=(1, 1), padding=(0, 0))

    def __call__(self, x, time):
        res = x
        x = self.group_norm_1(x)
        x = nn.silu(x)
        x = self.conv_1(x)
        time = nn.silu(time)
        time = self.time(time)
        m = self.conv_2(nn.silu(self.group_norm_2(x + time)))
        if self.idn:
            res = self.out(res)
        return m + res


class AttentionBlock(nn.Module):
    num_attention_heads: int
    hidden_size: int
    d_context: int = 768

    def setup(self) -> None:
        ch = self.num_attention_heads * self.hidden_size
        self.group_norm = nn.GroupNorm(32)
        self.conv_inp = nn.Conv(ch, kernel_size=(1, 1), padding=(1, 1))
        self.norm_1 = nn.LayerNorm()
        self.attn_1 = SelfAttention(features=ch, num_attention_heads=self.num_attention_heads)
        self.norm_2 = nn.LayerNorm()
        self.attn_2 = CrossAttention(features=ch, num_attention_heads=self.num_attention_heads)
        self.norm_3 = nn.LayerNorm()
        self.l1 = nn.Dense(ch * 4 * 2)
        self.l2 = nn.Dense(ch)
        self.out = nn.Conv(ch, kernel_size=(1, 1), padding=(0, 0))

    def __call__(self, x, context):
        residual = x
        x = self.group_norm(x)
        x = self.conv_inp(x)
        b, w, h, c = x.shape
        x = x.reshape(b, c, w * h)
        x = x.swapaxes(-1, -2)

        x = self.attn_1(self.norm_1(x)) + x
        x = self.attn_2(self.norm_2(x), context) + x

        res_ = x
        x = self.norm_3(x)
        x, gate = jnp.split(self.l1(x), axis=-1)
        x = x * nn.gelu(gate)
        x = self.l2(x)
        x = x + res_
        x = x.swapaxes(-1, -2)
        x = x.reshape(b, w, h, c)
        return self.out(x) + residual
