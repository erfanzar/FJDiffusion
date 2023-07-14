from transformers import PretrainedConfig
from typing import Tuple, Optional, Union
import jax.numpy as jnp
import jax
from jax.experimental.pjit import PartitionSpec


class AutoencoderKlConfig(PretrainedConfig):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 dropout_rate: float = 0.0,
                 epsilon: float = 1e-5,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 num_hidden_layers_per_block: int = 2,
                 sample_size: int = 256,
                 act_fn: str = "silu",
                 latent_channels: int = 4,
                 gradient_checkpointing: str = 'nothing_saveable',
                 **kwargs
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.num_hidden_layers_per_block = num_hidden_layers_per_block
        self.sample_size = sample_size
        self.act_fn = act_fn
        self.latent_channels = latent_channels
        self.gradient_checkpointing = gradient_checkpointing
        self.__dict__.update(**kwargs)

    def get_config_to_init(self, dtype: jnp.dtype = jnp.float32,
                           param_dtype: jnp.dtype = jnp.float32,
                           precision: Optional[Union[None, jax.lax.Precision]] = None):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "dropout_rate": self.dropout_rate,
            "epsilon": self.epsilon,
            "down_block_types": self.down_block_types,
            "up_block_types": self.up_block_types,
            "block_out_channels": self.block_out_channels,
            "num_hidden_layers_per_block": self.num_hidden_layers_per_block,
            "sample_size": self.sample_size,
            "act_fn": self.act_fn,
            "latent_channels": self.latent_channels,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": precision
        }

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = True):
        return (
            ("query/(kernel|bias)", PartitionSpec("fsdp")),
            ("c1/(bias|kernel)", PartitionSpec("fsdp")),
            ("value/(bias|kernel)", PartitionSpec("fsdp")),
            ("post_quant_conv/(bias|kernel)", PartitionSpec("fsdp")),
            ("out_norm/(scale|bias)", PartitionSpec("fsdp")),
            ("group_norm/(scale|bias)", PartitionSpec("fsdp")),
            ("conv_in/(bias|kernel)", PartitionSpec("fsdp")),
            ("norm_out/(scale|bias)", PartitionSpec("fsdp")),
            ("c2/(bias|kernel)", PartitionSpec("fsdp")),
            ("norm1/(scale|bias)", PartitionSpec("fsdp")),
            ("key/(bias|kernel)", PartitionSpec("fsdp")),
            ("proj_attn/(bias|kernel)", PartitionSpec("fsdp")),
            ("conv_out/(bias|kernel)", PartitionSpec("fsdp")),
            ("quant_conv/(bias|kernel)", PartitionSpec("fsdp")),
            ("norm2/(scale|bias)", PartitionSpec("fsdp")),
            ('.*', PartitionSpec(None))
        ) if fully_fsdp \
            else (
            ("query/(kernel|bias)", PartitionSpec("fsdp")),
            ("c1/(bias|kernel)", PartitionSpec("fsdp")),
            ("value/(bias|kernel)", PartitionSpec("fsdp")),
            ("post_quant_conv/(bias|kernel)", PartitionSpec("fsdp")),
            ("out_norm/(scale|bias)", PartitionSpec("fsdp")),
            ("group_norm/(scale|bias)", PartitionSpec("fsdp")),
            ("conv_in/(bias|kernel)", PartitionSpec("fsdp")),
            ("norm_out/(scale|bias)", PartitionSpec("fsdp")),
            ("c2/(bias|kernel)", PartitionSpec("fsdp")),
            ("norm1/(scale|bias)", PartitionSpec("fsdp")),
            ("key/(bias|kernel)", PartitionSpec("fsdp")),
            ("proj_attn/(bias|kernel)", PartitionSpec("fsdp")),
            ("conv_out/(bias|kernel)", PartitionSpec("fsdp")),
            ("quant_conv/(bias|kernel)", PartitionSpec("fsdp")),
            ("norm2/(scale|bias)", PartitionSpec("fsdp")),
            ('.*', PartitionSpec(None))
        )


class Unet2DConfig(PretrainedConfig):
    def __init__(self,
                 sample_size: int = 256,
                 in_channels: int = 4,
                 out_channels: int = 4,
                 down_block_types: Tuple[str, ...] = (
                         "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
                         "DownBlock2D"
                 ),
                 up_block_types: Tuple[str, ...] = (
                         "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
                         "CrossAttnUpBlock2D"
                 ),
                 only_cross_attention: Union[bool, Tuple[bool]] = False,
                 block_out_channels: Tuple[int, ...] = (320, 640, 640, 1280, 1280),
                 num_hidden_layers_per_block: int = 2,
                 dropout_rate: float = 0.0,
                 use_linear_proj: bool = False,
                 flip_sin_to_cos: bool = True,
                 num_attention_heads: Union[int, Tuple[int]] = 8,
                 cross_attention_dim: int = 1280,
                 freq_shift: int = 0,
                 gradient_checkpointing: str = 'nothing_saveable',
                 epsilon: float = 1e-5,
                 **kwargs
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.down_block_types = down_block_types
        self.up_block_types = up_block_types
        self.block_out_channels = block_out_channels
        self.only_cross_attention = only_cross_attention
        self.num_hidden_layers_per_block = num_hidden_layers_per_block
        self.flip_sin_to_cos = flip_sin_to_cos
        self.num_attention_heads = num_attention_heads
        self.use_linear_proj = use_linear_proj
        self.freq_shift = freq_shift
        self.cross_attention_dim = cross_attention_dim
        self.sample_size = sample_size
        self.gradient_checkpointing = gradient_checkpointing
        self.__dict__.update(**kwargs)

    def get_config_to_init(self, dtype: jnp.dtype = jnp.float32,
                           param_dtype: jnp.dtype = jnp.float32,
                           precision: Optional[Union[None, jax.lax.Precision]] = None):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "dropout_rate": self.dropout_rate,
            "epsilon": self.epsilon,
            "down_block_types": self.down_block_types,
            "up_block_types": self.up_block_types,
            "block_out_channels": self.block_out_channels,
            "only_cross_attention": self.only_cross_attention,
            "num_hidden_layers_per_block": self.num_hidden_layers_per_block,
            "flip_sin_to_cos": self.flip_sin_to_cos,
            "num_attention_heads": self.num_attention_heads,
            "use_linear_proj": self.use_linear_proj,
            "freq_shift": self.freq_shift,
            "cross_attention_dim": self.cross_attention_dim,
            "sample_size": self.sample_size,
            "gradient_checkpointing": self.gradient_checkpointing,
            "dtype": dtype,
            "param_dtype": param_dtype,
            "precision": precision
        }

    @staticmethod
    def get_partition_rules(fully_fsdp: bool = True):
        return (
            ("v/(kernel|bias)", PartitionSpec("fsdp")),
            ("q/(kernel|bias)", PartitionSpec("fsdp")),
            ("o/(kernel|bias)", PartitionSpec("fsdp")),
            ("k/(kernel|bias)", PartitionSpec("fsdp")),
            ("norm_out/(scale|bias)", PartitionSpec("fsdp")),
            ("time_emb/(kernel|bias)", PartitionSpec("fsdp")),
            ("conv_out/(kernel|bias)", PartitionSpec("fsdp")),
            ("proj/(kernel|bias)", PartitionSpec("fsdp")),
            ("c2/(kernel|bias)", PartitionSpec("fsdp")),
            ("l2/(kernel|bias)", PartitionSpec("fsdp")),
            ("proj_in/(kernel|bias)", PartitionSpec("fsdp")),
            ("cs/(kernel|bias)", PartitionSpec("fsdp")),
            ("c1/(kernel|bias)", PartitionSpec("fsdp")),
            ("net_2/(scale|bias)", PartitionSpec("fsdp")),
            ("norm2/(scale|bias)", PartitionSpec("fsdp")),
            ("conv_in/(kernel|bias)", PartitionSpec("fsdp")),
            ("norm1/(scale|bias)", PartitionSpec("fsdp")),
            ("norm3/(scale|bias)", PartitionSpec("fsdp")),
            ("l1/(kernel|bias)", PartitionSpec("fsdp")),
            ("conv/(kernel|bias)", PartitionSpec("fsdp")),
            ('.*', PartitionSpec(None))
        ) if fully_fsdp else (
            ("v/(kernel|bias)", PartitionSpec("fsdp")),
            ("q/(kernel|bias)", PartitionSpec("fsdp")),
            ("o/(kernel|bias)", PartitionSpec("fsdp")),
            ("k/(kernel|bias)", PartitionSpec("fsdp")),
            ("norm_out/(scale|bias)", PartitionSpec("fsdp")),
            ("time_emb/(kernel|bias)", PartitionSpec("fsdp")),
            ("conv_out/(kernel|bias)", PartitionSpec("fsdp")),
            ("proj/(kernel|bias)", PartitionSpec("fsdp")),
            ("c2/(kernel|bias)", PartitionSpec("fsdp")),
            ("l2/(kernel|bias)", PartitionSpec("fsdp")),
            ("proj_in/(kernel|bias)", PartitionSpec("fsdp")),
            ("cs/(kernel|bias)", PartitionSpec("fsdp")),
            ("c1/(kernel|bias)", PartitionSpec("fsdp")),
            ("net_2/(scale|bias)", PartitionSpec("fsdp")),
            ("norm2/(scale|bias)", PartitionSpec("fsdp")),
            ("conv_in/(kernel|bias)", PartitionSpec("fsdp")),
            ("norm1/(scale|bias)", PartitionSpec("fsdp")),
            ("norm3/(scale|bias)", PartitionSpec("fsdp")),
            ("l1/(kernel|bias)", PartitionSpec("fsdp")),
            ("conv/(kernel|bias)", PartitionSpec("fsdp")),
            ('.*', PartitionSpec(None))
        )


def get_clip_partition_rules(fully_fsdp: bool = True):
    return (
        ("token_embedding/embedding", PartitionSpec('mp', 'fsdp')),
        ("layer_norm1/(scale|bias)", PartitionSpec("fsdp")),
        ("fc2/(kernel|bias)", PartitionSpec("fsdp")),
        ("v_proj/(kernel|bias)", PartitionSpec("fsdp")),
        ("q_proj/(kernel|bias)", PartitionSpec("fsdp")),
        ("out_proj/(kernel|bias)", PartitionSpec("fsdp")),
        ("layer_norm2/(scale|bias)", PartitionSpec("fsdp")),
        ("position_embedding/embedding", PartitionSpec('mp', 'fsdp')),
        ("fc1/(kernel|bias)", PartitionSpec("fsdp")),
        ("final_layer_norm/(scale|bias)", PartitionSpec("fsdp")),
        ("k_proj/(kernel|bias)", PartitionSpec("fsdp")),
        ('.*', PartitionSpec(None))

    ) if fully_fsdp else (
        ("token_embedding/embedding", PartitionSpec('mp', 'fsdp')),
        ("layer_norm1/(scale|bias)", PartitionSpec('mp', "fsdp")),
        ("fc2/(kernel|bias)", PartitionSpec("fsdp", "mp")),
        ("v_proj/(kernel|bias)", PartitionSpec("fsdp", "mp")),
        ("q_proj/(kernel|bias)", PartitionSpec('mp', "fsdp")),
        ("out_proj/(kernel|bias)", PartitionSpec('mp', "fsdp")),
        ("layer_norm2/(scale|bias)", PartitionSpec('mp', "fsdp")),
        ("position_embedding/embedding", PartitionSpec('mp', 'fsdp')),
        ("fc1/(kernel|bias)", PartitionSpec("fsdp", "mp")),
        ("final_layer_norm/(scale|bias)", PartitionSpec("fsdp", "mp")),
        ("k_proj/(kernel|bias)", PartitionSpec('mp', "fsdp")),
        ('.*', PartitionSpec(None))
    )
