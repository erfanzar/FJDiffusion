from jax import numpy as jnp
from FJDiffusion.utils import get_alpha_cup
import numpy as np


class KEulerAncestralSampler:
    def __init__(self, n_inference_steps=50, n_training_steps=1000):
        timesteps = jnp.linspace(n_training_steps - 1, 0, n_inference_steps)

        alphas_cumprod = get_alpha_cup(training_steps=n_training_steps)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = jnp.log(sigmas)
        log_sigmas = jnp.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = jnp.exp(log_sigmas)
        sigmas = jnp.append(sigmas, 0)

        self.sigmas = sigmas
        self.initial_scale = sigmas.max()
        self.timesteps = timesteps
        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        self.step_count = 0

    def get_input_scale(self, step_count=None):
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        return 1 / (sigma ** 2 + 1) ** 0.5

    def set_strength(self, strength=1):
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = jnp.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
        self.timesteps = self.timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step

    def step(self, latents, output):
        t = self.step_count
        self.step_count += 1

        sigma_from = self.sigmas[t]
        sigma_to = self.sigmas[t + 1]
        sigma_up = sigma_to * (1 - (sigma_to ** 2 / sigma_from ** 2)) ** 0.5
        sigma_down = sigma_to ** 2 / sigma_from
        latents += output * (sigma_down - sigma_from)
        noise = jnp.asarray(np.random.randn(latents.shape).tolist())

        latents += noise * sigma_up
        return latents
