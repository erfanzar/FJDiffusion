import jax.lax

from FJDiffusion import AutoencoderKl, Unet2DConditionModel, Unet2DConfig, AutoencoderKlConfig
from .configs import get_clip_partition_rules
from transformers import FlaxCLIPTextModel, CLIPTokenizer, CLIPTextConfig
from typing import Optional, Tuple, Union
from jax import numpy as jnp
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
from fjutils.easylm import match_partition_rules
from FJDiffusion.utils import BaseClass


class MoonWalker(BaseClass):
    def __init__(
            self,
            unet_config_or_path: Union[Unet2DConfig, str],
            vae_config_or_path: Union[AutoencoderKlConfig, str],
            clip_config_or_path: Union[CLIPTextConfig, str],
            tokenizer_path: str,
            dtype: jnp.dtype = jnp.float32,
            param_dtype: jnp.dtype = jnp.float32,
            precision: Optional[Union[None, jax.lax.Precision]] = None,
            debug: Optional[bool] = False,
            clip_partition_rules: Optional[Union[None, tuple]] = None,
            fully_fsdp: bool = True,
            mesh_shape: Tuple[int, int, int] = (1, 1, -1, 1),
            backend: str = 'tpu'
    ):

        self.fully_fsdp = fully_fsdp
        self.mesh_shape = mesh_shape
        self.debug = debug
        self.backend = backend
        sharding_array = jnp.ones((len(jax.devices(backend)))).reshape(mesh_shape)
        self.sharding_shape = sharding_array.shape
        self.clip_partition_rules = clip_partition_rules

        config_vae = AutoencoderKlConfig.from_pretrained(vae_config_or_path) if isinstance(vae_config_or_path,
                                                                                           str) else vae_config_or_path
        config_clip = CLIPTextConfig.from_pretrained(clip_config_or_path) if isinstance(clip_config_or_path,
                                                                                        str) else clip_config_or_path
        config_unet = Unet2DConfig.from_pretrained(unet_config_or_path) if isinstance(unet_config_or_path,
                                                                                      str) else unet_config_or_path

        self.vae_partition = vae_config_or_path.get_partition_rules(fully_fsdp)
        self.unet_partition = unet_config_or_path.get_partition_rules(fully_fsdp)

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
        self.clip_init_shape = None
        self.vae_init_shape = None
        self.unet_init_shape = None
        self.clip_matched_partition_rules = None
        self.vae_matched_partition_rules = None
        self.unet_matched_partition_rules = None
        self.sharded_init_unet_params_func = None
        self.sharded_init_clip_params_func = None
        self.sharded_init_vae_params_func = None

        self.sharded_clip_params = RuntimeError(
            'You have to init the parameters with using MoonWalker.do_init function before calling clip_params')
        self.sharded_vae_params = RuntimeError(
            'You have to init the parameters with using MoonWalker.do_init function before calling vae_params')
        self.sharded_unet_params = RuntimeError(
            'You have to init the parameters with using MoonWalker.do_init function before calling unet_params')

        self.clip_params = RuntimeError(
            'You have to init the parameters with using MoonWalker.do_init function before calling clip_params')
        self.vae_params = RuntimeError(
            'You have to init the parameters with using MoonWalker.do_init function before calling vae_params')
        self.unet_params = RuntimeError(
            'You have to init the parameters with using MoonWalker.do_init function before calling unet_params')
        self.mesh = self.create_mesh()

    @classmethod
    def naming_mesh(cls):
        return 'dp', 'ddp', 'fsdp', 'mp',

    def create_mesh(self):

        physical_mesh = create_device_mesh(
            mesh_shape=self.sharding_shape
        )
        return Mesh(
            physical_mesh,
            self.naming_mesh()
        )

    def do_init(self, rng: jax.random.PRNGKey):
        clip_rng, unet_rng, vae_rng = jax.random.split(rng, 3)
        clip_partition = self.clip_partition_rules if self.clip_partition_rules is not None else \
            get_clip_partition_rules(self.fully_fsdp)
        vae_partition = self.vae_partition
        unet_partition = self.unet_partition

        def init_clip_params():
            clip_params = self.clip_model.init_weights(
                clip_rng, (1, 1)
            )
            return clip_params

        def init_vae_params():
            vae_params = self.vae_model.init_weights(
                vae_rng
            )
            return vae_params

        def init_unet_params():
            unet_params = self.unet_model.init_weights(
                unet_rng
            )
            return unet_params

        unet_init_shape = jax.eval_shape(
            init_unet_params
        )
        vae_init_shape = jax.eval_shape(
            init_vae_params
        )
        clip_init_shape = jax.eval_shape(
            init_clip_params
        )
        self.clip_init_shape = clip_init_shape
        self.vae_init_shape = vae_init_shape
        self.unet_init_shape = unet_init_shape

        self.clip_matched_partition_rules = match_partition_rules(clip_partition, clip_init_shape)
        self.vae_matched_partition_rules = match_partition_rules(vae_partition, vae_init_shape)
        self.unet_matched_partition_rules = match_partition_rules(unet_partition, unet_init_shape)

        self.sharded_init_clip_params_func = pjit(
            init_clip_params,
            out_shardings=self.clip_matched_partition_rules,
            # backend=self.backend
        )
        self.sharded_init_vae_params_func = pjit(
            init_vae_params,
            out_shardings=self.vae_matched_partition_rules,
            # backend=self.backend
        )
        self.sharded_init_unet_params_func = pjit(
            init_unet_params,
            out_shardings=self.unet_matched_partition_rules,
            # backend=self.backend
        )
        with self.mesh:
            sharded_clip_params = self.sharded_init_clip_params_func()

            if self.debug:
                print('CLIP Parameters initialized successfully')
            sharded_vae_params = self.sharded_init_vae_params_func()
            if self.debug:
                print('VAE Parameters initialized successfully')
            sharded_unet_params = self.sharded_init_unet_params_func()
            if self.debug:
                print('VAE Parameters initialized successfully')

        self.sharded_clip_params = sharded_clip_params
        self.sharded_vae_params = sharded_vae_params
        self.sharded_unet_params = sharded_unet_params
        self.clip_params = sharded_clip_params
        self.vae_params = sharded_vae_params
        self.unet_params = sharded_unet_params
