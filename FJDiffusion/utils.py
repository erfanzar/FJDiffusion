import jax.numpy as jnp
from flax import linen as nn
import jax


def pre_compute_time_embeddings(time_step, base: int = 10000, dim: int = 128, dtype: jnp.dtype = jnp.float32):
    freq = jnp.power(base, -jnp.arange(0, dim, dtype=dtype) / dim)
    x = jnp.arange([time_step], dtype=dtype)[:, jnp.newaxis] * freq[jnp.newaxis]
    return jnp.concatenate([jnp.cos(x), jnp.sin(x)], axis=-1)


def pre_compute_time_freq(base: int = 10000, dim: int = 64, max_length: int = 768, dtype: jnp.dtype = jnp.float32):
    freq = 1 / jnp.power(base, jnp.arange(0, dim, 2, dtype=dtype) / dim)
    t = jnp.arange(max_length)
    f = jnp.outer(t, freq)
    return jnp.sin(f).astype(dtype), jnp.cos(f).astype(dtype)


def get_alpha_cup(beta_start=0.00085, beta_end=0.0120, training_steps=1000, dtype: jnp.dtype = jnp.float32):
    betas = jnp.linspace(beta_start ** 0.5, beta_end ** 0.5, training_steps, dtype=dtype) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    return alphas_cumprod


class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.c = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.c(hidden_states)
        return hidden_states


class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.c = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        hidden_states = self.c(hidden_states)
        return hidden_states
