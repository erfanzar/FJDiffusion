import jax.numpy as jnp
import numpy as np
from typing import Callable


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


def preprocess_image(x):
    return ((x / 255) * 2) - 1


def past_process_image(x):
    return np.asarray(((x + 1) / 2) * 255, dtype=np.uint8)


class BaseClass:
    def __repr__(self):
        string = f'{self.__class__.__name__}('
        for k, v in self.__dict__.items():
            if isinstance(v, Callable):
                def string_func(it_self):

                    string_ = f'{it_self.__class__.__name__}(\n'
                    for k_, v_ in it_self.__dict__.items():
                        string_ += f'\t\t{k_} : {v_}\n'
                    string_ += '\t)'
                    return string_

                try:
                    v.__str__ = string_func
                    v = v.__str__(v)
                except RuntimeError:
                    pass

            string += f'\n\t{k} : {v}'
        string += ')'
        return string

    def __str__(self):
        return self.__repr__()
