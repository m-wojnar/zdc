import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, Reshape, UpSample


optimizer = optax.adam(1.4e-3, b1=0.76, b2=0.88, eps=6.1e-8, nesterov=True)


class ConvBlock(nn.Module):
    hidden_dim: int
    bottleneck_scale: int = 0.25

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.gelu(x)
        x = nn.Conv(int(self.hidden_dim * self.bottleneck_scale), kernel_size=(1, 1))(x)
        x = nn.gelu(x)
        x = nn.Conv(int(self.hidden_dim * self.bottleneck_scale), kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(int(self.hidden_dim * self.bottleneck_scale), kernel_size=(3, 3))(x)
        x = nn.gelu(x)
        x = nn.Conv(self.hidden_dim, kernel_size=(1, 1))(x)
        x = x + residual
        return x


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img, cond=None, training=True):
        x = img

        for features in [64, 128, 256]:
            x = nn.Conv(features, kernel_size=(1, 1))(x)
            x = ConvBlock(features)(x)
            x = ConvBlock(features)(x)
            x = nn.BatchNorm()(x, use_running_average=not training)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = Reshape((-1, 256))(x)
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

        x = nn.Dense(256)(z)
        x = Reshape((6, 6, 256))(x)

        for features in [256, 128, 64]:
            x = nn.Conv(features, kernel_size=(1, 1))(x)
            x = UpSample()(x)
            x = ConvBlock(features)(x)
            x = ConvBlock(features)(x)
            x = nn.BatchNorm()(x, use_running_average=not training)

        x = nn.Conv(1, kernel_size=(5, 5), padding='valid')(x)
        x = nn.relu(x + 0.5)
        return x
