import os
import typing

from . import CLIP, Diffusion, UnetModel, Decoder, Encoder
from transformers import CLIPTokenizer
from flax.core import FrozenDict

TYPE_CHECKING: bool = False


class FJDiffusion:
    decoder_params: FrozenDict = None
    encoder_params: FrozenDict = None
    diffusion_params: FrozenDict = None
    clip_params: FrozenDict = None

    def __init__(self,
                 tokenizer: CLIPTokenizer,
                 clip: CLIP,
                 diffusion: Diffusion,
                 decoder: Decoder,
                 encoder: Encoder,
                 clip_params: FrozenDict,
                 diffusion_params: FrozenDict,
                 decoder_params: FrozenDict,
                 encoder_params: FrozenDict
                 ):
        self.tokenizer = tokenizer

        self.clip = clip
        self.diffusion = diffusion
        self.decoder = decoder
        self.encoder = encoder

        self.clip_params = clip_params
        self.diffusion_params = diffusion_params
        self.encoder_params = encoder_params
        self.decoder_params = decoder_params

    @classmethod
    def load_from_ckpt(cls, path: typing.Union[os.PathLike, str], tokenizer: str):
        ...

    @classmethod
    def get_requirements(cls):
        return {
            'diffusion', 'decoder', 'encoder', 'clip'
        }

    @classmethod
    def get_status(cls):
        return {
            'diffusion': cls.diffusion_params,
            'decoder': cls.decoder_params,
            'encoder': cls.encoder_params,
            'clip': cls.clip_params,

        }

    def __call__(self, *args, **kwargs):
        ...

    def __del__(self):
        raise SystemError('Deletion is not supported for FJDiffusion (Create run time bug from jax CUDNn)')

    if TYPE_CHECKING:
        def __aexit__(self, exc_type, exc_val, exc_tb):
            ...
