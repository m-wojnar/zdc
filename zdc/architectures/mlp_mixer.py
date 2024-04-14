from functools import partial

import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, MixerBlock, Patches, PatchEncoder, Reshape
from zdc.utils.nn import opt_with_cosine_schedule


optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adamw, b1=0.89, b2=0.88, eps=1.2e-6, weight_decay=0.013),
    peak_value=5.7e-3,
    pct_start=0.34,
    div_factor=600,
    final_div_factor=1200,
    epochs=100,
    batch_size=256
)


class Encoder(nn.Module):
    hidden_dim: int = 96
    tokens_dim: int = 128
    channels_dim: int = 384
    num_layers: int = 5

    @nn.compact
    def __call__(self, img, cond=None, training=True):
        x = jnp.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x = Patches(patch_size=8)(x)
        x = PatchEncoder(x.shape[1], self.hidden_dim)(x)

        if cond is not None:
            c = nn.Dense(self.hidden_dim)(cond)
            c = Reshape((1, self.hidden_dim))(c)
            x = Concatenate(axis=1)(c, x)

        for _ in range(self.num_layers):
            x = MixerBlock(self.tokens_dim, self.channels_dim)(x)

        if cond is not None:
            x = x[:, 1:, :]

        x = Reshape((6 * 6, self.hidden_dim))(x)
        x = nn.LayerNorm()(x)

        return x


class Decoder(nn.Module):
    hidden_dim: int = 96
    tokens_dim: int = 128
    channels_dim: int = 384
    num_layers: int = 5

    @nn.compact
    def __call__(self, z, cond=None, training=True):
        x = PatchEncoder(z.shape[1], self.hidden_dim)(z)

        if cond is not None:
            c = nn.Dense(self.hidden_dim)(cond)
            c = Reshape((1, self.hidden_dim))(c)
            x = Concatenate(axis=1)(c, x)

        for _ in range(self.num_layers):
            x = MixerBlock(self.tokens_dim, self.channels_dim)(x)

        if cond is not None:
            x = x[:, 1:, :]

        x = Reshape((6, 6, self.hidden_dim))(x)
        x = nn.ConvTranspose(1, kernel_size=(8, 8), strides=(8, 8), padding='SAME')(x)
        x = nn.relu(x + 0.5)
        x = x[:, 2:-2, 2:-2, :]

        return x
