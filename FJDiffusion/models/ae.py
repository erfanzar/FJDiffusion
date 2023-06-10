from flax import linen as nn
from jax import numpy as jnp
from .attention import SelfAttentionWC, ResidualBlock
from .util_models import UpSampleFlax


class AttentionBlock(nn.Module):
    features: int

    def setup(self) -> None:
        super().__init__()
        self.groupnorm = nn.GroupNorm(32)
        self.attention = SelfAttentionWC(1, self.features)

    def forward(self, x):
        residue = x
        x = self.groupnorm(x)
        n, h, w, c = x.shape
        x = x.reshape((n, c, h * w)).swapaxes(-1, -2)
        x = self.attention(x).swapaxes(-1, -2).reshape((n, h, w, c)) + residue
        return x


class Encoder(nn.Sequential):
    feature_start: int = 128
    out_c: int = 8

    def setup(self) -> None:
        self.layers = [
            nn.Conv(self.feature_start, kernel_size=(3, 3), padding=(1, 1)),
            ResidualBlock(self.feature_start, True),
            ResidualBlock(self.feature_start, True),
            nn.Conv(self.feature_start, kernel_size=(3, 3), strides=(2, 2), padding=(0, 0)),
            ResidualBlock(self.feature_start * 2, False),
            ResidualBlock(self.feature_start * 2, True),
            nn.Conv(self.feature_start * 2, kernel_size=(3, 3), strides=(2, 2), padding=(0, 0)),
            ResidualBlock(self.feature_start * 4, False),
            ResidualBlock(self.feature_start * 4, True),
            nn.Conv(self.feature_start * 4, kernel_size=(3, 3), strides=(2, 2), padding=(0, 0)),
            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),
            AttentionBlock(self.feature_start * 4),
            ResidualBlock(self.feature_start * 4, True),
            nn.GroupNorm(32),
            nn.silu,
            nn.Conv(self.out_c, kernel_size=(3, 3), padding=(1, 1)),
            nn.Conv(self.out_c, kernel_size=(1, 1), padding=(0, 0)),
        ]

    def __call__(self, x, noise):
        for layer in self.layers:

            if getattr(layer, 'strides', None) == (2, 2):
                x = jnp.pad(x, (0, 1, 0, 1))
            x = layer(x)
        mean, log = jnp.split(x, 2, axis=-1)
        log = jnp.clip(log, -30, 20)
        log = jnp.sqrt(jnp.exp(log))
        x = mean + log * noise
        x *= 0.18215
        return x


class Decoder(nn.Sequential):
    feature_start: int = 128
    out_c: int = 3
    start_c: int = 4

    def setup(self) -> None:
        self.layers = [
            nn.Conv(self.start_c, kernel_size=(1, 1), padding=(0, 0)),
            nn.Conv(self.feature_start * 4, kernel_size=(3, 3), padding=(1, 1)),
            ResidualBlock(self.feature_start * 4, True),
            AttentionBlock(self.feature_start * 4),
            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),
            UpSampleFlax(2),
            nn.Conv(self.feature_start * 4, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),

            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),
            ResidualBlock(self.feature_start * 4, True),

            UpSampleFlax(2),
            nn.Conv(self.feature_start * 4, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            ResidualBlock(self.feature_start * 2, False),
            ResidualBlock(self.feature_start * 2, True),
            ResidualBlock(self.feature_start * 2, True),

            UpSampleFlax(2),
            nn.Conv(self.feature_start * 2, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1)),
            ResidualBlock(self.feature_start, False),
            ResidualBlock(self.feature_start, True),
            ResidualBlock(self.feature_start, True),
            nn.GroupNorm(32),
            nn.silu,
            nn.Conv(self.out_c, kernel_size=(3, 3), padding=(1, 1))

        ]

    def __call__(self, x):
        x /= 0.18215
        for layer in self.layers:
            x = layer(x)
        return x
