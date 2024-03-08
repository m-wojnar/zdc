import jax
import jax.numpy as jnp
from flax import linen as nn


class StochasticDepth(nn.Module):
    rate: float

    @nn.compact
    def __call__(self, x, training=True):
        if training and self.rate > 0.:
            keep_prob = 1. - self.rate
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + jax.random.uniform(self.make_rng('zdc'), shape, dtype=x.dtype)
            binary_tensor = jnp.floor(random_tensor)
            x = x / keep_prob * binary_tensor

        return x


class GlobalResponseNorm(nn.Module):
    dim: int = 1
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        gamma = self.param('gamma', nn.ones_init(), (1, 1, 1, self.dim))
        beta = self.param('beta', nn.zeros_init(), (1, 1, 1, self.dim))

        Gx = jnp.sqrt(jnp.square(x).sum(axis=(1, 2), keepdims=True))
        Nx = Gx / (Gx.mean(axis=-1, keepdims=True) + self.epsilon)
        return ((gamma * (x * Nx)) + beta) + x


class ConvNeXtV2Embedding(nn.Module):
    patch_size: int
    projection_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.projection_dim, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(x)
        x = nn.LayerNorm()(x)
        return x


class ConvNeXtV2Block(nn.Module):
    projection_dim: int
    kernel_size: int
    drop_rate: float = 0.

    @nn.compact
    def __call__(self, x, training=True):
        residual = x
        x = nn.Conv(self.projection_dim, kernel_size=(self.kernel_size, self.kernel_size), feature_group_count=self.projection_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(4 * self.projection_dim)(x)
        x = nn.gelu(x)
        x = GlobalResponseNorm()(x)
        x = nn.Dense(self.projection_dim)(x)
        x = StochasticDepth(self.drop_rate)(x, training=training)
        return x + residual


class ConvNeXtV2Stage(nn.Module):
    patch_size: int
    projection_dim: int
    kernel_size: int
    drop_rates: list

    @nn.compact
    def __call__(self, x, training=True):
        if self.projection_dim != x.shape[-1] or self.patch_size > 1:
            patch_size = (self.patch_size, self.patch_size)
            x = nn.LayerNorm()(x)
            x = nn.Conv(self.projection_dim, kernel_size=patch_size, strides=patch_size)(x)

        for drop_rate in self.drop_rates:
            x = ConvNeXtV2Block(self.projection_dim, self.kernel_size, drop_rate)(x, training=training)

        return x
