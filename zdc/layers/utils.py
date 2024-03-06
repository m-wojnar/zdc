import jax.numpy as jnp
from flax import linen as nn


class Reshape(nn.Module):
    shape: tuple

    @nn.compact
    def __call__(self, x):
        return x.reshape((x.shape[0],) + self.shape)


class Flatten(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x.reshape(x.shape[0], -1)


class Concatenate(nn.Module):
    axis: int = -1

    @nn.compact
    def __call__(self, *xs):
        return jnp.concatenate(xs, axis=self.axis)


class GlobalAveragePooling(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x.mean(axis=(1, 2))
