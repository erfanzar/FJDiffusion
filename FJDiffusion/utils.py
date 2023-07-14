import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable

jax_supported_dtypes = {
    'float64': jnp.float64,
    'float32': jnp.float32,
    'float16': jnp.float16,
    'bfloat16': jnp.bfloat16,
    'complex64': jnp.complex64
}


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
        return f'{self.__class__.__name__} {self.to_json()}'

    def __str__(self):
        return self.__repr__()

    def to_json(self):
        new_dict = {}
        for key, val in self.__dict__.items():
            if not isinstance(val, (list, tuple, int, float, dict, str)):
                try:
                    new_dict[key] = val.__class__.__name__
                except:
                    new_dict[key] = 'UnCatchAble'
            else:
                if isinstance(val, (list, tuple)):
                    new_dict[key] = val if len(val) < 8 else str(val[:4]) + '...'
                else:
                    new_dict[key] = val
        jsn = json.dumps(
            new_dict,
            indent=2,
            sort_keys=True
        )
        return jsn


def prefix_print(prefix, string):

    print(f"\033[1;32m{prefix}\033[1;0m : {string}")
