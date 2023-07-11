import typing

import jax
from jax import numpy as jnp
from flax import linen as nn


class FlaxResnetBlock2D(nn.Module):
    in_c: int
    out_c: int = None
    use_shortcut: bool = None
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        out_c = self.out_c or self.in_c
        self.c1 = nn.Conv(
            features=self.in_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm1 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )

        self.c2 = nn.Conv(
            features=out_c,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm2 = nn.GroupNorm(
            32, epsilon=self.epsilon
        )

        self.time_emb = nn.Dense(
            out_c,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.drop = nn.Dropout(
            self.dropout_rate
        )

        cut = self.in_c != out_c if self.use_shortcut is None else self.use_shortcut
        if cut:
            self.cs = nn.Conv(
                out_c,
                kernel_size=(1, 1),
                padding="VALID",
                strides=(1, 1),
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, time, deterministic=False):
        residual = hidden_state
        hidden_state = self.c1(nn.swish(self.norm1(hidden_state)))
        time = jnp.expand_dims(jnp.expand_dims(self.time_emb(nn.swish(time)), 1), 1)
        hidden_state += time
        hidden_state = self.c2(self.drop(nn.swish(self.norm2(hidden_state)), deterministic=deterministic))
        if self.cs is not None:
            residual = self.cs(residual)
        return hidden_state + residual
