from flax import linen as nn
from jax import numpy as jnp
from FJDiffusion.models.attention import SelfAttentionWC


class CLIP(nn.Module):
    vocab_size: int = 32000
    features: int = 768
    max_position_embedding: int = 128
    num_attention_heads: int = 12
    num_hidden_layers: int = 12

    def setup(self) -> None:
        self.embedding = CLIPEmbedding(
            vocab_size=self.vocab_size,
            features=self.features,
            max_position_embedding=self.max_position_embedding
        )
        self.ln = nn.LayerNorm()
        self.layers = [
            CLIPLayer(self.num_attention_heads, self.features) for _ in range(self.num_hidden_layers)
        ]

    def __call__(self, x):
        x = self.embedding(x.astype(dtype=jnp.int32))
        for layer in self.layers:
            x = layer(x)
        return self.ln(x)


class CLIPEmbedding(nn.Module):
    vocab_size: int = 32000
    features: int = 1280
    max_position_embedding: int = 256

    def setup(self) -> None:
        self.wte = nn.Embed(num_embeddings=self.vocab_size, features=self.features)
        self.pos = self.param('positions', nn.initializers.zeros, (self.max_position_embedding, self.features))

    def __call__(self, tokens):
        x = self.wte(tokens)
        x += self.pos
        return x


class CLIPLayer(nn.Module):
    num_attention_heads: int
    features: int

    def setup(self) -> None:
        self.ln1 = nn.LayerNorm()
        self.ln2 = nn.LayerNorm()
        self.l1 = nn.Dense(4 * self.features)
        self.l2 = nn.Dense(self.features)
        self.attn = SelfAttentionWC(num_attention_heads=self.num_attention_heads, features=self.features)

    def __call__(self, x):
        x = self.attn(self.ln1(x), True) + x
        residual = x
        x = self.l1(self.ln2(x))
        return self.l2(x * nn.sigmoid(1.702 * x)) + residual
