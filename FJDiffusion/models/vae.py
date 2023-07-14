import typing
from typing import Tuple

import jax
from flax import linen as nn
from jax import numpy as jnp

from FJDiffusion.moonwalker.utils import Downsample, Upsample
from FJDiffusion.models.resnet import FlaxResnetBlock2DNTime
from FJDiffusion.models.unet2d_blocks import FlaxUNetMidBlock2D
from FJDiffusion.models.utils import get_gradient_checkpointing_policy
import flax
from FJDiffusion.moonwalker.utils import BaseOutput


@flax.struct.dataclass
class FlaxAutoencoderKLOutput(BaseOutput):
    latent_dist: "FlaxDiagonalGaussianDistribution"


@flax.struct.dataclass
class FlaxDecoderOutput(BaseOutput):
    sample: jnp.ndarray


class FlaxDiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        """
        From huggingFace
        :param parameters:
        :param deterministic:
        """
        self.mean, self.logvar = jnp.split(parameters, 2, axis=-1)
        self.logvar = jnp.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = jnp.exp(0.5 * self.logvar)
        self.var = jnp.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)

    def kl(self, other=None):
        if self.deterministic:
            return jnp.array([0.0])

        if other is None:
            return 0.5 * jnp.sum(self.mean ** 2 + self.var - 1.0 - self.logvar, axis=[1, 2, 3])

        return 0.5 * jnp.sum(
            jnp.square(self.mean - other.mean) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar,
            axis=[1, 2, 3],
        )

    def nll(self, sample, axis=None):
        if axis is None:
            axis = [1, 2, 3]
        if self.deterministic:
            return jnp.array([0.0])

        logtwopi = jnp.log(2.0 * jnp.pi)
        return 0.5 * jnp.sum(logtwopi + self.logvar + jnp.square(sample - self.mean) / self.var, axis=axis)

    def mode(self):
        return self.mean


class FlaxDownEncoderBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.0
    add_down_sampler: bool = True
    epsilon: float = 1e-5
    num_hidden_layers: int = 1
    gradient_checkpointing: str = 'nothing_saveable'
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        resnet = []

        block_class = nn.remat(FlaxResnetBlock2DNTime, policy=get_gradient_checkpointing_policy(
            self.gradient_checkpointing)) if self.gradient_checkpointing != '' else FlaxResnetBlock2DNTime
        for i in range(self.num_hidden_layers):
            in_channel = self.in_channels if i == 0 else self.out_channels
            block = block_class(
                in_c=in_channel,
                out_c=self.out_channels,
                use_shortcut=None,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnet.append(block)

        self.resnets = resnet
        if self.add_down_sampler:
            self.down_sampler = Downsample(
                in_channel=self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, deterministic: bool = True):
        for block in self.resnets:
            hidden_state = block(
                hidden_state=hidden_state,
                deterministic=deterministic
            )
        if self.add_down_sampler:
            hidden_state = self.down_sampler(hidden_state)
        return hidden_state


class FlaxUpDecoderBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.0
    add_up_sampler: bool = True
    epsilon: float = 1e-5
    num_hidden_layers: int = 1
    gradient_checkpointing: str = 'nothing_saveable'
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        resnet = []

        block_class = nn.remat(FlaxResnetBlock2DNTime, policy=get_gradient_checkpointing_policy(
            self.gradient_checkpointing)) if self.gradient_checkpointing != '' else FlaxResnetBlock2DNTime
        for i in range(self.num_hidden_layers):
            in_channel = self.in_channels if i == 0 else self.out_channels
            block = block_class(
                in_c=in_channel,
                out_c=self.out_channels,
                use_shortcut=None,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnet.append(block)

        self.resnets = resnet
        if self.add_up_sampler:
            self.up_sampler = Upsample(
                in_channel=self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, deterministic: bool = True):
        for block in self.resnets:
            hidden_state = block(
                hidden_state=hidden_state,
                deterministic=deterministic
            )
        if self.add_up_sampler:
            hidden_state = self.up_sampler(hidden_state)
        return hidden_state


class FlaxDecoder(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    num_hidden_layers_per_block: int = 2
    gradient_checkpointing: str = 'nothing_saveable'
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: int = (64,)
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        self.conv_in = nn.Conv(
            self.block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.out_norm = nn.GroupNorm(
            32, epsilon=self.epsilon
        )
        self.bottle_neck = FlaxUNetMidBlock2D(
            in_channels=self.block_out_channels[-1],
            num_attention_heads=None,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            gradient_checkpointing=self.gradient_checkpointing
        )
        decoders = []
        block_class = nn.remat(FlaxUpDecoderBlock2D,
                               policy=get_gradient_checkpointing_policy(
                                   self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxUpDecoderBlock2D

        reversed_block_out_channels = list(self.block_out_channels[::-1])
        output_channel = reversed_block_out_channels[0]
        for i, name in enumerate(self.up_block_types):
            is_final_b = i == len(self.up_block_types) - 1
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            block = block_class(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                dropout_rate=self.dropout_rate,
                add_up_sampler=not is_final_b,
                epsilon=self.epsilon,
                num_hidden_layers=self.num_hidden_layers_per_block,
                gradient_checkpointing=self.gradient_checkpointing,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            decoders.append(block)
        self.decoders = decoders

    def __call__(self, hidden_state, deterministic: bool = True):
        hidden_state = self.bottle_neck(self.conv_in(hidden_state), deterministic=deterministic)
        for block in self.decoders:
            hidden_state = block(hidden_state=hidden_state, deterministic=deterministic)
        return self.conv_out(nn.swish(self.out_norm(hidden_state)))


class FlaxEncoder(nn.Module):
    in_channels: int
    out_channels: int
    double_z: bool = False
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    num_hidden_layers_per_block: int = 2
    gradient_checkpointing: str = 'nothing_saveable'
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    block_out_channels: int = (64,)
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        self.conv_in = nn.Conv(
            self.block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.conv_out = nn.Conv(
            self.out_channels * 2 if self.double_z else self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.norm_out = nn.GroupNorm(
            32, epsilon=self.epsilon
        )
        encoders = []
        out_c = self.block_out_channels[0]
        block_class = nn.remat(FlaxDownEncoderBlock2D,
                               policy=get_gradient_checkpointing_policy(
                                   self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxDownEncoderBlock2D
        for i, name in enumerate(self.down_block_types):
            in_c = out_c
            out_c = self.block_out_channels[i]
            is_final_b = i == len(self.down_block_types) - 1
            block = block_class(
                in_channels=in_c,
                out_channels=out_c,
                num_hidden_layers=self.num_hidden_layers_per_block,
                add_down_sampler=not is_final_b,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            encoders.append(block)
        self.encoders = encoders
        self.bottle_neck = FlaxUNetMidBlock2D(
            in_channels=self.block_out_channels[-1],
            num_attention_heads=None,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            gradient_checkpointing=self.gradient_checkpointing
        )

    def __call__(self, hidden_state, deterministic: bool = True):
        hidden_state = self.conv_in(hidden_state)
        for block in self.encoders:
            hidden_state = block(hidden_state=hidden_state, deterministic=deterministic)
        hidden_state = self.bottle_neck(hidden_states=hidden_state, deterministic=deterministic)
        return self.conv_out(nn.swish(self.norm_out(hidden_state)))


class AutoencoderKl(nn.Module):
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    down_block_types: Tuple[str] = ("DownEncoderBlock2D",)
    up_block_types: Tuple[str] = ("UpDecoderBlock2D",)
    block_out_channels: Tuple[int] = (64,)
    num_hidden_layers_per_block: int = 2
    sample_size: int = 32
    act_fn: str = "silu"
    latent_channels: int = 4
    gradient_checkpointing: str = 'nothing_saveable'
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None

    def setup(self) -> None:
        self.encoder = FlaxEncoder(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            double_z=True,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            num_hidden_layers_per_block=self.num_hidden_layers_per_block,
            gradient_checkpointing=self.gradient_checkpointing,
            down_block_types=self.down_block_types,
            block_out_channels=self.block_out_channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.decoder = FlaxDecoder(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            dropout_rate=self.dropout_rate,
            epsilon=self.epsilon,
            num_hidden_layers_per_block=self.num_hidden_layers_per_block,
            gradient_checkpointing=self.gradient_checkpointing,
            up_block_types=self.up_block_types,
            block_out_channels=self.block_out_channels,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.quant_conv = nn.Conv(
            2 * self.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='VALID',
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

        self.post_quant_conv = nn.Conv(
            self.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='VALID',
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )

    def init_weights(self, rng: jax.random.KeyArray):
        sample_shape = (1, self.in_channels, self.sample_size, self.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)

        params_rng, dropout_rng, gaussian_rng = jax.random.split(rng, 3)
        rngs = {"params": params_rng, "dropout": dropout_rng, "gaussian": gaussian_rng}

        return self.init(rngs, sample)["params"]

    def encode(self, sample, deterministic: bool = True, return_dict: bool = True):
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        hidden_state = self.quant_conv(self.encoder(sample, deterministic=deterministic))
        posterior = FlaxDiagonalGaussianDistribution(hidden_state, deterministic=False)
        if return_dict:
            return FlaxAutoencoderKLOutput(
                latent_dist=posterior
            )
        else:
            return posterior,

    def decode(self, latents, deterministic: bool = True, return_dict: bool = True):
        if latents.shape[-1] != self.latent_channels:
            latents = jnp.transpose(latents, (0, 2, 3, 1))

        hidden_states = self.post_quant_conv(latents)
        hidden_states = self.decoder(hidden_states, deterministic=deterministic)

        hidden_states = jnp.transpose(hidden_states, (0, 3, 1, 2))

        if not return_dict:
            return (hidden_states,)

        return FlaxDecoderOutput(sample=hidden_states)

    def __call__(self, sample, sample_posterior=False, deterministic: bool = True, return_dict: bool = True):
        posterior = self.encode(sample, deterministic=deterministic, return_dict=return_dict)
        if sample_posterior:
            rng = self.make_rng("gaussian")
            hidden_states = posterior.latent_dist.sample(rng)
        else:
            hidden_states = posterior.latent_dist.mode()

        sample = self.decode(hidden_states, return_dict=return_dict).sample

        if not return_dict:
            return (sample,)

        return FlaxDecoderOutput(sample=sample)
