import einops
import jax.nn.initializers
from flax import linen as nn
from jax import numpy as jnp
from FJDiffusion.models.attention import AttentionBlock, ResidualBlock


class UConv(nn.Module):
    features: int
    use_bias_bn: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        features = self.features
        self.c1 = nn.Conv(features=features, kernel_size=(3, 3), padding=1, dtype=self.dtype)
        self.c2 = nn.Conv(features=features, kernel_size=(3, 3), padding=1, dtype=self.dtype)
        self.b1 = nn.BatchNorm(epsilon=1e-6,
                               dtype=self.dtype,
                               use_running_average=False,
                               use_bias=self.use_bias_bn)
        self.b2 = nn.BatchNorm(epsilon=1e-6,
                               dtype=self.dtype,
                               use_running_average=False,
                               use_bias=self.use_bias_bn)

    def __call__(self, x):
        return nn.relu(self.b2(self.c2(nn.relu(self.b1(self.c1(x))))))


class EBConv(nn.Module):
    features: int

    def setup(self) -> None:
        self.encoder = UConv(features=self.features)

    def __call__(self, x):
        x = self.encoder(x)
        p = nn.max_pool(x, (2, 2), strides=(2, 2))
        return x, p


class DBConv(nn.Module):
    features: int

    def setup(self) -> None:
        self.decoder = nn.ConvTranspose(features=self.features, kernel_size=(2, 2), strides=(2, 2), padding=(1, 1))
        self.block = UConv(features=self.features)

    def __call__(self, x, skip):
        x = self.decoder(x)
        x = jnp.concatenate([x, skip], axis=-1)
        return self.block(x)


class SwitchSequential(nn.Sequential):
    def __call__(self, x, context, time):
        if not self.layers:
            raise ValueError(f'Empty Sequential module {self.name}.')
        for layer in self.layers:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UpSampleConv(nn.Module):
    features: int

    def setup(self) -> None:
        self.conv = nn.Conv(self.features, kernel_size=(3, 3), padding=(1, 1))

    def __call__(self, x):
        if x.ndim == 3:
            height, weight, _ = x.shape
        elif x.ndim == 4:
            _, height, weight, _ = x.shape
        else:
            raise ValueError
        x = jax.image.resize(x, (height * 2, weight * 2), method='nearest')
        return self.conv(x)


class UpSampleFlax(nn.Module):
    factor_size: int = 2
    method: str = 'bilinear'

    def __call__(self, x):
        if x.ndim == 3:
            height, weight, _ = x.shape
        elif x.ndim == 4:
            _, height, weight, _ = x.shape
        else:
            raise ValueError
        return jax.image.resize(x, (height * self.factor_size, weight * self.factor_size), method=self.method)
