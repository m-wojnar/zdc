from functools import partial

import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, Patches, PatchEncoder, Reshape, TransformerBlock
from zdc.utils.nn import opt_with_cosine_schedule


optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adamw, b1=0.64, b2=0.73, eps=1.5e-7, weight_decay=0.068),
    peak_value=3.5e-3,
    pct_start=0.32,
    div_factor=35,
    final_div_factor=770,
    epochs=100,
    batch_size=256
)


class Encoder(nn.Module):
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1

    @nn.compact
    def __call__(self, img, cond=None, training=True):
        x = jnp.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x = Patches(patch_size=8)(x)
        x = PatchEncoder(x.shape[1], self.hidden_dim, positional_encoding=True)(x)

        if cond is not None:
            c = nn.Dense(self.hidden_dim)(cond)
            c = Reshape((1, self.hidden_dim))(c)
            x = Concatenate(axis=1)(c, x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate)(x, training=training)

        if cond is not None:
            x = x[:, 1:, :]

        x = Reshape((6 * 6, self.hidden_dim))(x)
        x = nn.LayerNorm()(x)

        return x


class Decoder(nn.Module):
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1

    @nn.compact
    def __call__(self, z, cond=None, training=True):
        x = PatchEncoder(z.shape[1], self.hidden_dim, positional_encoding=True)(z)

        if cond is not None:
            c = nn.Dense(self.hidden_dim)(cond)
            c = Reshape((1, self.hidden_dim))(c)
            x = Concatenate(axis=1)(c, x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate)(x, training=training)

        if cond is not None:
            x = x[:, 1:, :]

        x = Reshape((6, 6, self.hidden_dim))(x)
        x = nn.ConvTranspose(1, kernel_size=(8, 8), strides=(8, 8), padding='SAME')(x)
        x = nn.relu(x + 0.5)
        x = x[:, 2:-2, 2:-2, :]

        return x
