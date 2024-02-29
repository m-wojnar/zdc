import jax.numpy as jnp
from flax import linen as nn


class Reshape(nn.Module):
    shape: tuple

    @nn.compact
    def __call__(self, x):
        return x.reshape(self.shape)


class Concatenate(nn.Module):
    axis: int

    @nn.compact
    def __call__(self, x1, x2):
        return jnp.concatenate([x1, x2], axis=self.axis)
