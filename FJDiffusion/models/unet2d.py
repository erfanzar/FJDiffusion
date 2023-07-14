import dataclasses

import flax.linen as nn
import flax.struct
import jax.lax
import jax.numpy as jnp
from typing import Optional, Union, Tuple
from FJDiffusion.models.unet2d_blocks import FlaxCrossAttnUpBlock, FlaxCrossAttnDownBlock, FlaxUpBlock2D, \
    FlaxDownBlock2D, FlaxUNetMidBlock2DCrossAttn
from FJDiffusion.moonwalker.utils import FlaxTimesteps, FlaxTimestepEmbedding, BaseOutput


@flax.struct.dataclass
class OutputType(BaseOutput):
    sample: jnp.array


class Unet2DConditionModel(nn.Module):
    sample_size: int = 32
    in_channels: int = 4
    out_channels: int = 4

    down_block_types: Tuple[str] = (
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"
    )
    up_block_types: Tuple[str] = (
        "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
    )
    only_cross_attention: Union[bool, Tuple[bool]] = False
    block_out_channels: Tuple[int] = (320, 640, 640, 1280, 1280)
    num_hidden_layers_per_block: int = 2
    dropout_rate: float = 0.0
    use_linear_proj: bool = False

    flip_sin_to_cos: bool = True
    num_attention_heads: Union[int, Tuple[int]] = 8
    cross_attention_dim: int = 1280
    freq_shift: int = 0

    gradient_checkpointing: str = 'nothing_saveable'

    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    precision: Optional[Union[None, jax.lax.Precision]] = None

    epsilon: float = 1e-5

    def init_weights(self, rng: jax.random.KeyArray):

        sample = jnp.zeros((1, self.in_channels, self.sample_size, self.sample_size), dtype=self.dtype)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim), dtype=self.dtype)

        params_rng, dropout_rng = jax.random.split(rng)

        return self.init({"params": params_rng, "dropout": dropout_rng}, sample, timesteps, encoder_hidden_states)[
            "params"]

    def setup(self) -> None:
        assert len(self.down_block_types) == len(self.up_block_types)
        time_embedding_dimension = self.block_out_channels[0] * 4
        num_attention_heads = self.num_attention_heads
        self.conv_in = nn.Conv(
            self.block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.time_o = FlaxTimesteps(
            self.block_out_channels[0], flip_sin_to_cos=self.flip_sin_to_cos, freq_shift=self.freq_shift
        )
        self.time_e = FlaxTimestepEmbedding(
            time_embedding_dimension,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        only_cross_atn = self.only_cross_attention
        if isinstance(only_cross_atn, bool):
            only_cross_atn = [only_cross_atn] * len(
                self.down_block_types
            )
        if isinstance(num_attention_heads, int):
            num_attention_heads = [num_attention_heads] * len(self.down_block_types)
        output_channel = self.block_out_channels[0]
        down_blocks = []
        for i, name in enumerate(self.down_block_types):
            in_channels = output_channel
            output_channel = self.block_out_channels[i]
            is_final_b = i == len(self.down_block_types) - 1
            if name == 'CrossAttnDownBlock2D':
                block = FlaxCrossAttnDownBlock(
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    gradient_checkpointing=self.gradient_checkpointing,
                    epsilon=self.epsilon,
                    dropout_rate=self.dropout_rate,
                    num_attention_heads=num_attention_heads[i],
                    use_linear_proj=self.use_linear_proj,
                    add_downsampler=not is_final_b,
                    num_hidden_layers=self.num_hidden_layers_per_block,
                    in_channels=in_channels,
                    only_cross_attn=only_cross_atn[i],
                    out_channels=output_channel,
                )
            elif name == 'DownBlock2D':
                block = FlaxDownBlock2D(
                    in_channels=in_channels,
                    out_channels=output_channel,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    add_downsampler=not is_final_b,
                    num_hidden_layers=self.num_hidden_layers_per_block,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    gradient_checkpointing=self.gradient_checkpointing
                )
            else:
                raise ValueError(
                    f'{name} in down block types are not valid valid down blocks are DownBlock2D, CrossAttnDownBlock2D')
            down_blocks.append(block)
        self.bottle_neck = FlaxUNetMidBlock2DCrossAttn(
            in_channels=self.block_out_channels[-1],
            dropout_rate=self.dropout_rate,
            num_attention_heads=num_attention_heads[-1],
            use_linear_proj=self.use_linear_proj,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            gradient_checkpointing=self.gradient_checkpointing
        )
        up_blocks = []
        reversed_block_out_channels = list(self.block_out_channels[::-1])
        reversed_num_attention_heads = list(num_attention_heads[::-1])
        reversed_only_cross_atn = list(only_cross_atn[::-1])
        output_channel = reversed_block_out_channels[0]
        for i, name in enumerate(self.up_block_types):
            perv_output_channel = output_channel
            in_channels = reversed_block_out_channels[min(i + 1, len(self.block_out_channels) - 1)]
            output_channel = reversed_block_out_channels[i]
            is_final_b = i == len(reversed_block_out_channels) - 1
            if name == 'CrossAttnUpBlock2D':
                block = FlaxCrossAttnUpBlock(
                    in_channels=in_channels,
                    out_channels=output_channel,
                    perv_out_channels=perv_output_channel,
                    num_attention_heads=reversed_num_attention_heads[i],
                    num_hidden_layers=self.num_hidden_layers_per_block,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    only_cross_attn=reversed_only_cross_atn[i],
                    use_linear_proj=self.use_linear_proj,
                    add_upsampler=not is_final_b,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    gradient_checkpointing=self.gradient_checkpointing
                )
            elif name == 'UpBlock2D':
                block = FlaxUpBlock2D(
                    in_channels=in_channels,
                    out_channels=output_channel,
                    perv_out_channels=perv_output_channel,
                    num_attention_heads=reversed_num_attention_heads[i],
                    num_hidden_layers=self.num_hidden_layers_per_block,
                    dropout_rate=self.dropout_rate,
                    epsilon=self.epsilon,
                    only_cross_attn=reversed_only_cross_atn[i],
                    use_linear_proj=self.use_linear_proj,
                    add_upsampler=not is_final_b,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    precision=self.precision,
                    gradient_checkpointing=self.gradient_checkpointing,
                )
            else:
                raise ValueError(
                    f'{name} in up block types are not valid valid down blocks are UpBlock2D, CrossAttnUpBlock2D')

            up_blocks.append(block)

        self.norm_out = nn.GroupNorm(32, epsilon=self.epsilon)
        self.conv_out = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision
        )
        self.up_blocks = up_blocks
        self.down_blocks = down_blocks

    def __call__(self,
                 hidden_states,
                 timestep,
                 encoder_hidden_states,
                 down_block_additional_residuals=None,
                 bottle_neck_additional_residuals=None,
                 return_dict: bool = True,
                 deterministic: bool = True
                 ):
        if not isinstance(timestep, jnp.ndarray):
            timestep = jnp.array([timestep], dtype=jnp.int32)
        elif isinstance(timestep, jnp.ndarray) and len(timestep.shape) == 0:
            timestep = jnp.expand_dims(timestep.astype(dtype=jnp.float32), 0)

        time = self.time_e(self.time_o(timestep))
        hidden_states = self.conv_in(jnp.transpose(hidden_states, (0, 2, 3, 1)))
        down_block_res_hidden_states = [hidden_states]
        for block in self.down_blocks:
            if isinstance(block, FlaxCrossAttnDownBlock):
                hidden_states, res_hidden_states = block(hidden_state=hidden_states,
                                                         time=time,
                                                         encoder_hidden_states=encoder_hidden_states,
                                                         deterministic=deterministic
                                                         )
            elif isinstance(block, FlaxDownBlock2D):
                hidden_states, res_hidden_states = block(
                    hidden_state=hidden_states, time=time, deterministic=deterministic
                )
            else:
                raise RuntimeError()
            down_block_res_hidden_states.append(res_hidden_states)
        if down_block_additional_residuals is not None:
            new_down_block_res_hidden_states = []

            for down_block_res_hidden_states, down_block_additional_residual in zip(
                    down_block_res_hidden_states, down_block_additional_residuals
            ):
                new_down_block_res_hidden_states.append(down_block_res_hidden_states + down_block_additional_residual)

            down_block_res_hidden_states = new_down_block_res_hidden_states
        hidden_states = self.bottle_neck(
            hidden_states=hidden_states,
            time=time,
            encoder_hidden_states=encoder_hidden_states,
            deterministic=deterministic
        )
        if bottle_neck_additional_residuals is not None:
            hidden_states += bottle_neck_additional_residuals

        for block in self.up_blocks:
            res_hidden_states = down_block_res_hidden_states.pop()
            if isinstance(block, FlaxCrossAttnUpBlock):
                hidden_states = block(
                    hidden_state=hidden_states,
                    time=time,
                    deterministic=deterministic,
                    encoder_hidden_states=encoder_hidden_states,
                    output_states=res_hidden_states
                )
            elif isinstance(block, FlaxUpBlock2D):
                hidden_states = block(
                    hidden_state=hidden_states,
                    time=time,
                    output_states=res_hidden_states,
                    deterministic=deterministic
                )
            else:
                raise RuntimeError()

        hidden_states = jnp.transpose(self.conv_out(nn.swish(self.norm_out(hidden_states))), (0, 3, 1, 2))
        if not return_dict:
            return hidden_states,
        else:
            return OutputType(sample=hidden_states)
