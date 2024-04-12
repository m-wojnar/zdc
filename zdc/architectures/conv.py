import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, ConvBlock, Reshape, UpSample
from zdc.utils.nn import opt_with_cosine_schedule


optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img, cond=None, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = Reshape((-1, 128))(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        if cond is not None:
            cond = jnp.repeat(cond[:, None], x.shape[1], axis=1)
            x = Concatenate()(x, cond)

        return x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, cond=None, training=True):
        if cond is not None:
            cond = jnp.repeat(cond[:, None], z.shape[1], axis=1)
            z = Concatenate()(z, cond)

        x = nn.Dense(128)(z)
        x = Reshape((6, 6, 128))(x)
        x = UpSample()(x)
        x = ConvBlock(128, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(64, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(32, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = nn.Conv(1, kernel_size=(5, 5), padding='valid')(x)
        x = nn.relu(x)
        return x
