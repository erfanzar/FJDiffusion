from jax import numpy as jnp
import jax
from flax import linen as nn
from FJDiffusion.models.attn import FlaxTransformerBlock2D, FlaxAttentionBlock
from FJDiffusion.models.resnet import FlaxResnetBlock2D, FlaxResnetBlock2DNTime
from FJDiffusion.moonwalker.utils import Downsample, Upsample
from FJDiffusion.models.utils import get_gradient_checkpointing_policy
import typing


class FlaxCrossAttnDownBlock(nn.Module):
    in_channels: int
    out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_downsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnets = []
        attentions = []
        resnet_block = nn.remat(FlaxResnetBlock2D,
                                policy=get_gradient_checkpointing_policy(
                                    name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxResnetBlock2D

        attention_block = nn.remat(FlaxTransformerBlock2D,
                                   policy=get_gradient_checkpointing_policy(
                                       name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxTransformerBlock2D

        for index in range(self.num_hidden_layers):
            in_channels = self.in_channels if index == 0 else self.out_channels
            res_n = resnet_block(
                in_c=in_channels,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            atn_n = attention_block(
                num_hidden_layers=1,
                heads_dim=self.out_channels // self.num_attention_heads,
                num_attention_heads=self.num_attention_heads,
                use_linear_proj=self.use_linear_proj,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                only_cross_attn=self.only_cross_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            attentions.append(atn_n)
            resnets.append(res_n)
        self.attentions = attentions
        self.resnets = resnets
        if self.add_downsampler:
            self.downsamplers_0 = Downsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, time, encoder_hidden_states, deterministic: bool = True):
        output_states = ()
        for resnet, attention in zip(self.resnets, self.attentions):
            hidden_state = resnet(hidden_state=hidden_state, time=time, deterministic=deterministic)
            hidden_state = attention(hidden_state=hidden_state, context=encoder_hidden_states,
                                     deterministic=deterministic)
            output_states += (hidden_state,)
        if self.add_downsampler:
            hidden_state = self.downsamplers_0(hidden_state)
            output_states += (hidden_state,)
        return hidden_state, output_states


class FlaxDownBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    add_downsampler: bool = False
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnet = []
        resnet_block = nn.remat(FlaxResnetBlock2D,
                                policy=get_gradient_checkpointing_policy(
                                    name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxResnetBlock2D
        for index in range(self.num_hidden_layers):
            in_channels = self.in_channels if index == 0 else self.out_channels
            res_n = resnet_block(
                in_c=in_channels,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

            resnet.append(res_n)

        self.resnets = resnet
        if self.add_downsampler:
            self.downsamplers_0 = Downsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self, hidden_state, time, deterministic: bool = True):
        output_states = []
        for resnet in self.resnets:
            hidden_state = resnet(hidden_state=hidden_state, time=time, deterministic=deterministic)
            output_states.append(hidden_state)
        if self.add_downsampler:
            hidden_state = self.downsamplers_0(hidden_state)
            output_states.append(hidden_state)
        return hidden_state, output_states


class FlaxCrossAttnUpBlock(nn.Module):
    in_channels: int
    out_channels: int
    perv_out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 2
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_upsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnet = []
        attentions = []
        resnet_block = nn.remat(FlaxResnetBlock2D,
                                policy=get_gradient_checkpointing_policy(
                                    name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxResnetBlock2D

        attention_block = nn.remat(FlaxTransformerBlock2D,
                                   policy=get_gradient_checkpointing_policy(
                                       name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxTransformerBlock2D

        for index in range(self.num_hidden_layers):
            in_channel = self.in_channels if (index == self.num_hidden_layers - 1) else self.out_channels
            resnet_skip_in_channel = self.perv_out_channels if index == 0 else self.out_channels
            res_n = resnet_block(
                in_c=in_channel + resnet_skip_in_channel,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnet.append(res_n)
            atn_n = attention_block(
                num_hidden_layers=1,
                heads_dim=self.out_channels // self.num_attention_heads,
                num_attention_heads=self.num_attention_heads,
                use_linear_proj=self.use_linear_proj,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                only_cross_attn=self.only_cross_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            attentions.append(atn_n)

        self.resnets = resnet
        self.attentions = attentions
        if self.add_upsampler:
            self.upsampler = Upsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self,
                 hidden_state: jnp.DeviceArray,
                 time: jnp.DeviceArray,
                 output_states: list,
                 encoder_hidden_states: jnp.DeviceArray,
                 deterministic: bool = True
                 ):
        output_states = tuple(output_states)
        for res, atn in zip(self.resnets, self.attentions):
            enc = output_states[-1]
            output_states = output_states[:-1]
            hidden_state = jnp.concatenate([hidden_state, enc], axis=-1)
            hidden_state = res(hidden_state, time, deterministic=deterministic)
            hidden_state = atn(hidden_state, encoder_hidden_states, deterministic=deterministic)
        if self.add_upsampler:
            hidden_state = self.upsampler(hidden_state)
        return hidden_state


class FlaxUpBlock2D(nn.Module):
    in_channels: int
    out_channels: int
    perv_out_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_upsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self) -> None:
        resnet = []
        resnet_block = nn.remat(FlaxResnetBlock2D,
                                policy=get_gradient_checkpointing_policy(
                                    name=self.gradient_checkpointing)) \
            if self.gradient_checkpointing != '' else FlaxResnetBlock2D

        for index in range(self.num_hidden_layers):
            in_channel = self.in_channels if (index == self.num_hidden_layers - 1) else self.out_channels
            resnet_skip_in_channel = self.perv_out_channels if index == 0 else self.out_channels
            res_n = resnet_block(
                in_c=in_channel + resnet_skip_in_channel,
                out_c=self.out_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnet.append(res_n)

        self.resnets = resnet
        if self.add_upsampler:
            self.upsampler = Upsample(
                self.out_channels,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )

    def __call__(self,
                 hidden_state: jnp.DeviceArray,
                 time: jnp.DeviceArray,
                 output_states: list,
                 deterministic: bool = True
                 ):
        output_states = tuple(output_states)
        for res in self.resnets:
            enc = output_states[-1]
            output_states = output_states[:-1]
            hidden_state = jnp.concatenate([hidden_state, enc], axis=-1)
            hidden_state = res(hidden_state, time, deterministic=deterministic)

        if self.add_upsampler:
            hidden_state = self.upsampler(hidden_state)
        return hidden_state


class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    in_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    only_cross_attn: bool = False
    use_linear_proj: bool = False
    add_upsampler: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self):
        attention_block = nn.remat(
            FlaxTransformerBlock2D,
            policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
        )
        resnet_block = nn.remat(
            FlaxResnetBlock2D,
            policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
        )
        resnets = [
            resnet_block(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        ]

        attentions = []

        for _ in range(self.num_hidden_layers):
            attn_block = attention_block(
                num_hidden_layers=1,
                heads_dim=self.in_channels // self.num_attention_heads,
                num_attention_heads=self.num_attention_heads,
                use_linear_proj=self.use_linear_proj,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                only_cross_attn=self.only_cross_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision,
                gradient_checkpointing=self.gradient_checkpointing
            )
            attentions.append(attn_block)

            res_block = resnet_block(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 time: jnp.DeviceArray,
                 encoder_hidden_states: jnp.DeviceArray,
                 deterministic=True
                 ):
        hidden_states = self.resnets[0](hidden_states, time)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
            hidden_states = resnet(hidden_states, time, deterministic=deterministic)

        return hidden_states


class FlaxUNetMidBlock2D(nn.Module):
    in_channels: int
    num_attention_heads: int
    num_hidden_layers: int = 1
    dropout_rate: float = 0.0
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: typing.Optional[typing.Union[None, jax.lax.Precision]] = None
    gradient_checkpointing: str = 'nothing_saveable'

    def setup(self):
        attention_block = nn.remat(
            FlaxAttentionBlock,
            policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
        ) if self.gradient_checkpointing != '' else FlaxAttentionBlock
        resnet_block = nn.remat(
            FlaxResnetBlock2DNTime,
            policy=get_gradient_checkpointing_policy(self.gradient_checkpointing)
        ) if self.gradient_checkpointing != '' else FlaxResnetBlock2DNTime
        resnets = [
            resnet_block(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
        ]

        attentions = []

        for _ in range(self.num_hidden_layers):
            attn_block = attention_block(
                channels=self.in_channels,
                num_attention_heads=self.num_attention_heads,
                num_groups=32,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            attentions.append(attn_block)

            res_block = resnet_block(
                in_c=self.in_channels,
                out_c=self.in_channels,
                dropout_rate=self.dropout_rate,
                epsilon=self.epsilon,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                precision=self.precision
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self,
                 hidden_states: jnp.DeviceArray,
                 deterministic=True
                 ):
        hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states)
            hidden_states = resnet(hidden_states, deterministic=deterministic)

        return hidden_states
