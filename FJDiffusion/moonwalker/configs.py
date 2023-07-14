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
        return (("query/kernel", PartitionSpec("fsdp")),
                ("c1/bias", PartitionSpec("fsdp")),
                ("value/bias", PartitionSpec("fsdp")),
                ("post_quant_conv/kernel", PartitionSpec("fsdp")),
                ("value/kernel", PartitionSpec("fsdp")),
                ("out_norm/scale", PartitionSpec("fsdp")),
                ("out_norm/bias", PartitionSpec("fsdp")),
                ("group_norm/scale", PartitionSpec("fsdp")),
                ("post_quant_conv/bias", PartitionSpec("fsdp")),
                ("conv_in/kernel", PartitionSpec("fsdp")),
                ("norm_out/bias", PartitionSpec("fsdp")),
                ("c2/kernel", PartitionSpec("fsdp")),
                ("quant_conv/bias", PartitionSpec("fsdp")),
                ("norm_out/scale", PartitionSpec("fsdp")),
                ("norm1/bias", PartitionSpec("fsdp")),
                ("key/kernel", PartitionSpec("fsdp")),
                ("proj_attn/kernel", PartitionSpec("fsdp")),
                ("norm2/bias", PartitionSpec("fsdp")),
                ("conv_out/kernel", PartitionSpec("fsdp")),
                ("key/bias", PartitionSpec("fsdp")),
                ("c1/kernel", PartitionSpec("fsdp")),
                ("quant_conv/kernel", PartitionSpec("fsdp")),
                ("conv_out/bias", PartitionSpec("fsdp")),
                ("group_norm/bias", PartitionSpec("fsdp")),
                ("query/bias", PartitionSpec("fsdp")),
                ("conv_in/bias", PartitionSpec("fsdp")),
                ("c2/bias", PartitionSpec("fsdp")),
                ("norm1/scale", PartitionSpec("fsdp")),
                ("proj_attn/bias", PartitionSpec("fsdp")),
                ("norm2/scale", PartitionSpec("fsdp"))) if fully_fsdp \
            else (("query/kernel", PartitionSpec("fsdp", "mp")),
                  ("c1/bias", PartitionSpec("mp", "fsdp")),
                  ("value/bias", PartitionSpec("mp", "fsdp")),
                  ("post_quant_conv/kernel", PartitionSpec("fsdp")),
                  ("value/kernel", PartitionSpec("fsdp", "mp")),
                  ("out_norm/scale", PartitionSpec("fsdp", "mp")),
                  ("out_norm/bias", PartitionSpec("fsdp", "mp")),
                  ("group_norm/scale", PartitionSpec("fsdp")),
                  ("post_quant_conv/bias", PartitionSpec("fsdp")),
                  ("conv_in/kernel", PartitionSpec("fsdp", "mp")),
                  ("norm_out/bias", PartitionSpec("fsdp")),
                  ("c2/kernel", PartitionSpec("fsdp")),
                  ("quant_conv/bias", PartitionSpec("fsdp", "mp")),
                  ("norm_out/scale", PartitionSpec("fsdp")),
                  ("norm1/bias", PartitionSpec("fsdp", "mp")),
                  ("key/kernel", PartitionSpec("fsdp", "mp")),
                  ("proj_attn/kernel", PartitionSpec("fsdp", "mp")),
                  ("norm2/bias", PartitionSpec("mp", "fsdp")),
                  ("conv_out/kernel", PartitionSpec("fsdp", "mp")),
                  ("key/bias", PartitionSpec("mp", "fsdp")),
                  ("c1/kernel", PartitionSpec("fsdp", "mp")),
                  ("quant_conv/kernel", PartitionSpec("fsdp")),
                  ("conv_out/bias", PartitionSpec("fsdp", "mp")),
                  ("group_norm/bias", PartitionSpec("fsdp")),
                  ("query/bias", PartitionSpec("fsdp", "mp")),
                  ("conv_in/bias", PartitionSpec("fsdp", "mp")),
                  ("c2/bias", PartitionSpec("mp", "fsdp")),
                  ("norm1/scale", PartitionSpec("fsdp", "mp")),
                  ("proj_attn/bias", PartitionSpec("fsdp", "mp")),
                  ("norm2/scale", PartitionSpec("mp", "fsdp")))


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
                 num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
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
            ("norm_out/scale", PartitionSpec("fsdp")),
            ("conv_in/kernel", PartitionSpec("fsdp")),
            ("conv/bias", PartitionSpec("fsdp")),
            ("l1/bias", PartitionSpec("fsdp")),
            ("norm3/bias", PartitionSpec("fsdp")),
            ("time_emb/bias", PartitionSpec("fsdp")),
            ("v/kernel", PartitionSpec("fsdp")),
            ("conv_out/kernel", PartitionSpec("fsdp")),
            ("k/kernel", PartitionSpec("fsdp")),
            ("proj/kernel", PartitionSpec("fsdp")),
            ("c2/kernel", PartitionSpec("fsdp")),
            ("l2/bias", PartitionSpec("fsdp")),
            ("proj_in/bias", PartitionSpec("fsdp")),
            ("q/kernel", PartitionSpec("fsdp")),
            ("cs/kernel", PartitionSpec("fsdp")),
            ("conv_out/bias", PartitionSpec("fsdp")),
            ("c1/bias", PartitionSpec("fsdp")),
            ("time_emb/kernel", PartitionSpec("fsdp")),
            ("net_2/bias", PartitionSpec("fsdp")),
            ("norm2/scale", PartitionSpec("fsdp")),
            ("proj_in/kernel", PartitionSpec("fsdp")),
            ("conv_in/bias", PartitionSpec("fsdp")),
            ("norm1/bias", PartitionSpec("fsdp")),
            ("norm3/scale", PartitionSpec("fsdp")),
            ("l2/kernel", PartitionSpec("fsdp")),
            ("l1/kernel", PartitionSpec("fsdp")),
            ("c1/kernel", PartitionSpec("fsdp")),
            ("norm1/scale", PartitionSpec("fsdp")),
            ("conv/kernel", PartitionSpec("fsdp")),
            ("o/bias", PartitionSpec("fsdp")),
            ("proj/bias", PartitionSpec("fsdp")),
            ("norm2/bias", PartitionSpec("fsdp")),
            ("cs/bias", PartitionSpec("fsdp")),
            ("net_2/kernel", PartitionSpec("fsdp")),
            ("norm_out/bias", PartitionSpec("fsdp")),
            ("c2/bias", PartitionSpec("fsdp")),
            ("o/kernel", PartitionSpec("fsdp")),
            ('.*', PartitionSpec(None))
        ) if fully_fsdp else (
            ("norm_out/scale", PartitionSpec("fsdp", "mp")),
            ("conv_in/kernel", PartitionSpec("fsdp", "mp")),
            ("conv/bias", PartitionSpec("mp", "fsdp")),
            ("l1/bias", PartitionSpec("fsdp", "mp")),
            ("norm3/bias", PartitionSpec("fsdp", "mp")),
            ("time_emb/bias", PartitionSpec("mp", "fsdp")),
            ("v/kernel", PartitionSpec("mp", "fsdp")),
            ("conv_out/kernel", PartitionSpec("fsdp", "mp")),
            ("k/kernel", PartitionSpec("fsdp", "mp")),
            ("proj/kernel", PartitionSpec("fsdp")),
            ("c2/kernel", PartitionSpec("fsdp", "mp")),
            ("l2/bias", PartitionSpec("fsdp", "mp")),
            ("proj_in/bias", PartitionSpec("fsdp")),
            ("q/kernel", PartitionSpec("fsdp")),
            ("cs/kernel", PartitionSpec("fsdp", "mp")),
            ("conv_out/bias", PartitionSpec("fsdp")),
            ("c1/bias", PartitionSpec("fsdp", "mp")),
            ("time_emb/kernel", PartitionSpec("fsdp")),
            ("net_2/bias", PartitionSpec("fsdp")),
            ("norm2/scale", PartitionSpec("fsdp")),
            ("proj_in/kernel", PartitionSpec("fsdp", "mp")),
            ("conv_in/bias", PartitionSpec("fsdp")),
            ("norm1/bias", PartitionSpec("fsdp", "mp")),
            ("norm3/scale", PartitionSpec("fsdp", "mp")),
            ("l2/kernel", PartitionSpec("mp", "fsdp")),
            ("l1/kernel", PartitionSpec("fsdp", "mp")),
            ("c1/kernel", PartitionSpec("fsdp", "mp")),
            ("norm1/scale", PartitionSpec("mp", "fsdp")),
            ("conv/kernel", PartitionSpec("fsdp", "mp")),
            ("o/bias", PartitionSpec("mp", "fsdp")),
            ("proj/bias", PartitionSpec("mp", "fsdp")),
            ("norm2/bias", PartitionSpec("mp", "fsdp")),
            ("cs/bias", PartitionSpec("fsdp", "mp")),
            ("net_2/kernel", PartitionSpec("fsdp", "mp")),
            ("norm_out/bias", PartitionSpec("mp", "fsdp")),
            ("c2/bias", PartitionSpec("mp", "fsdp")),
            ("o/kernel", PartitionSpec("mp", "fsdp")),
            ('.*', PartitionSpec(None))
        )


def get_clip_partition_rules(fully_fsdp: bool = True):
    return (
        ("fc2/bias", PartitionSpec("fsdp")),
        ("layer_norm1/bias", PartitionSpec("fsdp")),
        ("fc1/kernel", PartitionSpec("fsdp")),
        ("final_layer_norm/bias", PartitionSpec("fsdp")),
        ("token_embedding/embedding", PartitionSpec("fsdp")),
        ("layer_norm1/scale", PartitionSpec("fsdp")),
        ("fc2/kernel", PartitionSpec("fsdp")),
        ("v_proj/kernel", PartitionSpec("fsdp")),
        ("q_proj/bias", PartitionSpec("fsdp")),
        ("q_proj/kernel", PartitionSpec("fsdp")),
        ("out_proj/kernel", PartitionSpec("fsdp")),
        ("out_proj/bias", PartitionSpec("fsdp")),
        ("layer_norm2/bias", PartitionSpec("fsdp")),
        ("layer_norm2/scale", PartitionSpec("fsdp")),
        ("position_embedding/embedding", PartitionSpec("fsdp")),
        ("v_proj/bias", PartitionSpec("fsdp")),
        ("fc1/bias", PartitionSpec("fsdp")),
        ("final_layer_norm/scale", PartitionSpec("fsdp")),
        ("k_proj/bias", PartitionSpec("fsdp")),
        ("k_proj/kernel", PartitionSpec("fsdp")),
        ('.*', PartitionSpec(None))

    ) if fully_fsdp else (
        ("fc2/bias", PartitionSpec("fsdp", "mp")),
        ("layer_norm1/bias", PartitionSpec("fsdp", "mp")),
        ("fc1/kernel", PartitionSpec("mp", "fsdp")),
        ("final_layer_norm/bias", PartitionSpec("fsdp", "mp")),
        ("token_embedding/embedding", PartitionSpec("fsdp", "mp")),
        ("layer_norm1/scale", PartitionSpec("mp", "fsdp")),
        ("fc2/kernel", PartitionSpec("mp", "fsdp")),
        ("v_proj/kernel", PartitionSpec("fsdp", "mp")),
        ("q_proj/bias", PartitionSpec("fsdp", "mp")),
        ("q_proj/kernel", PartitionSpec("mp", "fsdp")),
        ("out_proj/kernel", PartitionSpec("mp", "fsdp")),
        ("out_proj/bias", PartitionSpec("mp", "fsdp")),
        ("layer_norm2/bias", PartitionSpec("fsdp", "mp")),
        ("layer_norm2/scale", PartitionSpec("fsdp", "mp")),
        ("position_embedding/embedding", PartitionSpec("mp", "fsdp")),
        ("v_proj/bias", PartitionSpec("mp", "fsdp")),
        ("fc1/bias", PartitionSpec("fsdp", "mp")),
        ("final_layer_norm/scale", PartitionSpec("mp", "fsdp")),
        ("k_proj/bias", PartitionSpec("fsdp", "mp")),
        ("k_proj/kernel", PartitionSpec("mp", "fsdp")),
        ('.*', PartitionSpec(None))
    )
