from flax import linen as nn
from jax import numpy as jnp
from FJDiffusion.cdiffusion import CDiffusion
from FJDiffusion.models import UnetModel


class TTIDiffusion(nn.Module):
    vocab_size: int
    image_size: int
    feature_start: int = 128
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def setup(self) -> None:
        self.wte = nn.Embed(
            self.vocab_size,
            self.feature_start,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.unet = UnetModel(
            feature_start=self.feature_start,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.diff = CDiffusion(
            noise_steps=self.noise_steps,
            beta_end=self.beta_end,
            beta_start=self.beta_start,
            img_size=self.image_size
        )

    def __call__(self, input_ids):
        ...
