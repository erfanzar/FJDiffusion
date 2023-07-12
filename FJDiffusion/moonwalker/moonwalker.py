import jax.lax

from FJDiffusion import AutoencoderKl, Unet2DConditionModel, Unet2DConfig, AutoencoderKlConfig
from transformers import FlaxCLIPTextModel, CLIPTokenizer, CLIPTextConfig
from typing import Optional, Union
from jax import numpy as jnp


class MoonWalker:
    def __init__(
            self,
            unet_config_or_path: Union[Unet2DConfig, str],
            vae_config_or_path: Union[AutoencoderKlConfig, str],
            clip_config_or_path: Union[CLIPTextConfig, str],
            tokenizer_path: str,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[Union[None, jax.lax.Precision]] = None
    ):
        config_vae = AutoencoderKlConfig.from_pretrained(vae_config_or_path) if isinstance(vae_config_or_path,
                                                                                           str) else vae_config_or_path
        config_clip = CLIPTextConfig.from_pretrained(clip_config_or_path) if isinstance(clip_config_or_path,
                                                                                        str) else clip_config_or_path
        config_unet = Unet2DConfig.from_pretrained(unet_config_or_path) if isinstance(unet_config_or_path,
                                                                                      str) else unet_config_or_path

        config_unet_kwargs = config_unet.get_config_to_init()
        config_vae_kwargs = config_vae.get_config_to_init()
        config_unet_kwargs['dtype'] = dtype
        config_unet_kwargs['param_dtype'] = param_dtype
        config_unet_kwargs['precision'] = precision
        config_vae_kwargs['dtype'] = dtype
        config_vae_kwargs['param_dtype'] = param_dtype
        config_vae_kwargs['precision'] = precision

        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        clip_model = FlaxCLIPTextModel(
            config=config_clip, dtype=dtype, _do_init=False
        )

        unet_model = Unet2DConditionModel(
            **config_unet_kwargs
        )
        vae_model = AutoencoderKl(
            **config_vae_kwargs
        )
        self.vae_model = vae_model
        self.unet_model = unet_model
        self.clip_model = clip_model
        self.tokenizer = tokenizer
