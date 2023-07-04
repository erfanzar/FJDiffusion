import jax
from jax import numpy as jnp
import numpy as np
from flax import linen as nn


class CDiffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.prepare_noise_schedule()
        self.alpha = 1. - self.beta
        self.alpha_hat = jnp.cumprod(self.alpha, axis=0)

    def prepare_noise_schedule(self):
        return jnp.linspace(
            start=self.beta_start ** 0.5,
            stop=self.beta_end ** 0.5,
            num=self.noise_steps
        ) ** 2

    def noise_images(self, inp, time):
        sqrt_alpha_hat = jnp.sqrt(self.alpha_hat[time]).reshape(1, 1)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        sqrt_om_alpha_hat = jnp.sqrt(1 - self.alpha_hat[time]).reshape(1, 1)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
        e = np.random.randn(*inp.shape)
        return (sqrt_alpha_hat * inp) + (sqrt_om_alpha_hat * e), e

    def sample_time_steps(self, key: jax.random.PRNGKey, number):
        return jax.random.randint(key, minval=1, maxval=self.noise_steps, shape=(number,))

    def __call__(self, model: nn.Module, params, number):
        try:
            params = params['params']
        except KeyError:
            params = params
        x = jnp.asarray(np.random.randn((number, self.img_size, self.img_size, 3)).tolist())
        for i in reversed(range(1, self.noise_steps)):
            time = (jnp.ones(number) * i).astype(jnp.int32)
            pred = model.apply({'params': params}, x=x, time=time)
            alpha = self.alpha[time][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            alpha_hat = self.alpha_hat[time][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            beta = self.beta[time][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            noise = jnp.asarray(np.random.randn(*x.shape)) if i > 1 else jnp.zeros_like(x)
            x = 1 / jnp.sqrt(alpha) * (x - ((1 - alpha) / (jnp.sqrt(1 - alpha_hat))) * pred) + jnp.sqrt(
                beta
            ) * noise
        return (((x.clip(-1, 1) + 1) / 2) * 255).astype(jnp.uint8)
