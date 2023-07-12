from jax import numpy as jnp
import jax
from flax import linen as nn
from FJDiffusion.transformers.resnet import FlaxResnetBlock2DNTime
from FJDiffusion.moonwalker.utils import Downsample, Upsample
from FJDiffusion.transformers.utils import get_gradient_checkpointing_policy
from FJDiffusion.transformers.unet2d_blocks import FlaxUNetMidBlock2D
import typing
from typing import Optional, Union, Tuple
from FJDiffusion.transformers.attn import FlaxBaseAttn


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

    def __call__(self, hidden_state, deterministic: bool = True):
        ...


class FlaxEncoder(nn.Module):
    def setup(self) -> None:
        ...

    def __call__(self, *args, **kwargs):
        ...
