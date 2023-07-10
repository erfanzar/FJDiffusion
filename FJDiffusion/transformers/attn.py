import typing

import jax
from jax import numpy as jnp
from flax import linen as nn


class FlaxBaseAttn(nn.Module):
    query_dim: int
    num_attention_heads: int = 8
    heads_dim: int = 64
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        iner_dim = self.heads_dim * self.num_attention_heads
        self.q = nn.Dense(iner_dim,
                          dtype=self.dtype,
                          pram_dtype=self.param_dtype,
                          precision=self.precision)
        self.k = nn.Dense(iner_dim,
                          dtype=self.dtype,
                          pram_dtype=self.param_dtype,
                          precision=self.precision)
        self.v = nn.Dense(iner_dim,
                          dtype=self.dtype,
                          pram_dtype=self.param_dtype,
                          precision=self.precision)
        self.o = nn.Dense()
        self.dropout_layer = nn.Dropout(rate=self.dropout)

    def __call__(self, *args, **kwargs):
        ...
