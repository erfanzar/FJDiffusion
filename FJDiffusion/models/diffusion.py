from flax import linen as nn
from jax import numpy as jnp

from FJDiffusion.models.util_models import SwitchSequential, UpSampleConv
from FJDiffusion.models.attention import ResidualBlock, AttentionBlock


class UnetModel(nn.Module):
    feature_start: int
    param_dtype: jnp.dtype = jnp.float32
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.encoder = [
            SwitchSequential([nn.Conv(self.feature_start, kernel_size=(3, 3), padding=(1, 1))]),
            SwitchSequential([ResidualBlock(self.feature_start, True), AttentionBlock(8, 40)]),
            SwitchSequential([ResidualBlock(self.feature_start, True), AttentionBlock(8, 40)]),
            SwitchSequential([nn.Conv(self.feature_start, kernel_size=(3, 3), padding=(1, 1), strides=(2, 2))]),
            SwitchSequential([ResidualBlock(self.feature_start * 2, False), AttentionBlock(8, 80)]),
            SwitchSequential([ResidualBlock(self.feature_start * 2, True), AttentionBlock(8, 80)]),
            SwitchSequential([nn.Conv(self.feature_start * 2, kernel_size=(3, 3), padding=(1, 1), strides=(2, 2))]),
            SwitchSequential([ResidualBlock(self.feature_start * 4, False), AttentionBlock(8, 160)]),
            SwitchSequential([ResidualBlock(self.feature_start * 4, True), AttentionBlock(8, 160)]),
            SwitchSequential([nn.Conv(self.feature_start * 4, kernel_size=(3, 3), padding=(1, 1), strides=(2, 2))]),
            SwitchSequential([ResidualBlock(self.feature_start * 4, True)]),
            SwitchSequential([ResidualBlock(self.feature_start * 4, True)]),
        ]
        self.bottle_neck = SwitchSequential(
            [
                ResidualBlock(1280),
                AttentionBlock(8, 160),
                ResidualBlock(1280)
            ]
        )

        self.decoder = [
            SwitchSequential([ResidualBlock(self.feature_start * 4)]),
            SwitchSequential([ResidualBlock(self.feature_start * 4)]),
            SwitchSequential([ResidualBlock(self.feature_start * 4), UpSampleConv(self.feature_start * 4)]),
            SwitchSequential([ResidualBlock(self.feature_start * 4), AttentionBlock(8, 160)]),
            SwitchSequential([ResidualBlock(self.feature_start * 4), AttentionBlock(8, 160)]),
            SwitchSequential(
                [ResidualBlock(self.feature_start * 4), AttentionBlock(8, 160), UpSampleConv(self.feature_start * 4)]),
            SwitchSequential([ResidualBlock(self.feature_start * 2), AttentionBlock(8, 80)]),
            SwitchSequential([ResidualBlock(self.feature_start * 2), AttentionBlock(8, 80)]),
            SwitchSequential(
                [ResidualBlock(self.feature_start * 2), AttentionBlock(8, 80), UpSampleConv(self.feature_start * 2)]),
            SwitchSequential([ResidualBlock(self.feature_start), AttentionBlock(8, 40)]),
            SwitchSequential([ResidualBlock(self.feature_start), AttentionBlock(8, 40)]),
            SwitchSequential([ResidualBlock(self.feature_start), AttentionBlock(8, 40)]),
        ]

    def __call__(self, x, time, context=None):
        context = x if context is None else context

        route = []
        for layers in self.encoder:
            x = layers(x, context=context, time=time)
            route.append(x)
        x = self.bottle_neck(x, context=context, time=time)
        for layers in self.decoder:
            x = jnp.concatenate([x, route.pop()], dim=-1)
            x = layers(x=x, context=context, time=time)
        return x


class TE(nn.Module):
    features: int

    def setup(self) -> None:
        self.d1 = nn.Dense(4 * self.features)
        self.d2 = nn.Dense(self.features * 4)

    def __call__(self, x):
        return self.d2(nn.silu(self.d1(x)))


class Diffusion(nn.Module):
    in_channels: int = 4
    feature_start: int = 320

    def setup(self) -> None:
        self.group_norm = nn.GroupNorm()
        self.conv = nn.Conv(self.in_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_embedding = TE(self.feature_start)
        self.unet = UnetModel(self.feature_start)

    def __call__(self, latent, context, time):
        time = self.time_embedding(time)
        o = self.conv(nn.silu(self.group_norm(self.unet(latent, context, time))))
        return o
