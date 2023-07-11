import typing

import jax
from jax import numpy as jnp
from flax import linen as nn


def get_gradient_checkpointing_policy(name):
    return {
        "everything_saveable": jax.checkpoint_policies.everything_saveable,
        "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
        "dots_saveable": jax.checkpoint_policies.dots_saveable,
        "checkpoint_dots": jax.checkpoint_policies.dots_saveable,
        "dots_with_no_batch_dims_saveable": jax.checkpoint_policies.dot_with_no_batch_dims_saveable,
        "checkpoint_dots_with_no_batch_dims": jax.checkpoint_policies.dot_with_no_batch_dims_saveable,
        "save_anything_except_these_names": jax.checkpoint_policies.save_anything_except_these_names,
        "save_any_names_but_these": jax.checkpoint_policies.save_any_names_but_these,
        "save_only_these_names": jax.checkpoint_policies.save_only_these_names,
        "save_from_both_policies": jax.checkpoint_policies.save_from_both_policies
    }[name]


class FlaxBaseAttn(nn.Module):
    query_dim: int
    num_attention_heads: int = 8
    heads_dim: int = 64
    dropout_rate: float = 0.0
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        inner_dim = self.heads_dim * self.num_attention_heads
        self.scale = self.heads_dim ** -0.5
        self.q = nn.Dense(inner_dim,
                          dtype=self.dtype,
                          pram_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=jax.nn.initializers.normal(),
                          use_bias=False)
        self.k = nn.Dense(inner_dim,
                          dtype=self.dtype,
                          pram_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=jax.nn.initializers.normal(),
                          use_bias=False)
        self.v = nn.Dense(inner_dim,
                          dtype=self.dtype,
                          pram_dtype=self.param_dtype,
                          precision=self.precision,
                          kernel_init=jax.nn.initializers.normal(),
                          use_bias=False)
        self.o = nn.Dense(
            self.query_dim,
            dtype=self.dtype,
            pram_dtype=self.param_dtype,
            precision=self.precision,
            kernel_init=jax.nn.initializers.normal(),
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def split(self, x: jnp.DeviceArray):
        batch, sq, hidden_size = x.shape
        x = x.reshape(batch, sq, self.num_attention_heads, hidden_size // self.num_attention_heads)
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch * self.num_attention_heads, sq, hidden_size * self.num_attention_heads)

    def merge(self, x: jnp.DeviceArray):
        batch, sq, hidden_size = x.shape
        x = x.reshape(batch // self.num_attention_heads, self.num_attention_heads, sq, hidden_size)
        x = jnp.transpose(x, (0, 2, 1, 3))
        return x.reshape(batch // self.num_attention_heads, sq, self.num_attention_heads * hidden_size)

    def __call__(self, hidden_state: jnp.DeviceArray,
                 context: typing.Optional[typing.Union[None, jnp.DeviceArray]] = None,
                 deterministic: bool = False):
        context = hidden_state if context is None else context
        q = self.q(hidden_state)
        v = self.v(context)
        k = self.k(context)
        q, k, v = self.split(q), self.split(k), self.split(v)
        attn = jax.nn.softmax(jnp.einsum('b i d,b j d-> b i j', q, k) * self.scale, axis=-1)
        attn = self.merge(jnp.einsum('b i j,b j d -> b i d', attn, v))
        return self.dropout(self.o(attn), deterministic=deterministic)


class FlaxFeedForward(nn.Module):
    features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        self.net_0 = FlaxGEGLU(features=self.features,
                               dropout_rate=self.dropout_rate,
                               dtype=self.dtype,
                               param_dtype=self.param_dtype,
                               precision=self.precision)
        self.net_2 = nn.Dense(self.features,
                              dtype=self.dtype,
                              param_dtype=self.param_dtype,
                              precision=self.precision)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states, deterministic=deterministic)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self):
        inner_features = self.features * 4
        self.proj = nn.Dense(inner_features * 2,
                             dtype=self.dtype,
                             param_dtype=self.param_dtype,
                             precision=self.precision)
        self.dropout_layer = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return self.dropout_layer(hidden_linear * nn.gelu(hidden_gelu), deterministic=deterministic)


class FlaxEncoderBaseTransformerBlock(nn.Module):
    features: int
    num_attention_heads: int
    heads_dim: int
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        self.attn_1 = FlaxBaseAttn(
            self.features,
            num_attention_heads=self.num_attention_heads,
            dim_head=self.heads_dim,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.attn_2 = FlaxBaseAttn(
            self.features,
            num_attention_heads=self.num_attention_heads,
            dropout_rate=self.dropout_rate,
            dim_head=self.heads_dim,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.ffd = FlaxFeedForward(
            features=self.features,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            dropout_rate=self.dropout_rate,
            precision=self.precision
        )
        self.norm1 = nn.LayerNorm(epsilon=self.epsilon,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype)
        self.norm2 = nn.LayerNorm(epsilon=self.epsilon,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype)
        self.norm3 = nn.LayerNorm(epsilon=self.epsilon,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def __call__(self, hidden_state: jnp.DeviceArray, context: jnp.DeviceArray, deterministic: bool = True):

        if self.only_cross_attn:
            hidden_state = self.attn_1(self.norm1(hidden_state), context=context,
                                       deterministic=deterministic) + hidden_state
        else:
            hidden_state = self.attn_1(self.norm1(hidden_state), context=None,
                                       deterministic=deterministic) + hidden_state

        hidden_state = self.attn_2(self.norm2(hidden_state), context, deterministic=deterministic) + hidden_state
        hidden_state = self.ffd(self.norm3(hidden_state), deterministic=deterministic) + hidden_state
        return self.dropout_layer(hidden_state, deterministic=deterministic)


class FlaxEncoderBaseTransformerBlockCollection(nn.Module):
    features: int
    num_attention_heads: int
    heads_dim: int
    num_hidden_layers: int
    dropout_rate: float
    epsilon: float
    only_cross_attn: bool
    gradient_checkpointing: str
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        block = FlaxEncoderBaseTransformerBlock
        if self.gradient_checkpointing != '':
            block = nn.remat(
                block,
                policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
            )
        self.blocks = [
            block(
                features=self.features,
                heads_dim=self.heads_dim,
                num_attention_heads=self.num_attention_heads,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                name=str(i)
            ) for i in range(self.num_hidden_layers)
        ]

    def __call__(self, hidden_state: jnp.DeviceArray, context: jnp.DeviceArray, deterministic: bool = True):
        for block in self.blocks:
            hidden_state = block(
                hidden_state=hidden_state,
                context=context,
                deterministic=deterministic
            )
        return hidden_state


class FlaxTransformerBlock2D(nn.Module):
    in_channels: int
    num_attention_heads: int
    heads_dim: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        features = self.heads_dim * self.num_attention_heads
        self.norm1 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )
        if self.use_linear_proj:
            self.proj_in = nn.Dense(
                features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        else:
            self.proj_in = nn.Conv(
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='VALID',
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        self.blocks = FlaxEncoderBaseTransformerBlockCollection(
            features=features,
            num_attention_heads=self.num_attention_heads,
            heads_dim=self.heads_dim,
            num_hidden_layers=self.num_hidden_layers,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            only_cross_attn=self.only_cross_attn,
            gradient_checkpointing=self.gradient_checkpointing
        )
        if self.use_linear_proj:
            self.proj_out = nn.Dense(
                features,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        else:
            self.proj_out = nn.Conv(
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='VALID',
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        self.dropout_layer = nn.Dropout(
            self.dropout_rate
        )

    def __call__(self, hidden_state: jnp.DeviceArray, context: jnp.DeviceArray, deterministic: bool = True):
        batch, height, width, channels = hidden_state.shape
        residual = hidden_state
        hidden_state = self.norm1(hidden_state)
        if self.use_linear_proj:
            hidden_state = self.proj_in(
                hidden_state.reshape(batch, height * width, channels)
            )
        else:
            hidden_state = self.proj_in(
                hidden_state
            ).reshape(batch, height * width, channels)
        hidden_state = self.blocks(hidden_state=hidden_state, context=context, deterministic=deterministic)
        if self.use_linear_proj:

            hidden_state = self.proj_out(
                hidden_state
            ).reshape(batch, height, width, channels)
        else:
            hidden_state = self.proj_in(
                hidden_state.reshape(batch, height, width, channels)
            )
        return self.dropout_layer(hidden_state + residual, deterministic=deterministic)
