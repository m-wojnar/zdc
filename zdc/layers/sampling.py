import jax
import jax.numpy as jnp
from flax import linen as nn


class Sampling(nn.Module):
    @nn.compact
    def __call__(self, z_mean, z_log_var):
        epsilon = jax.random.normal(self.make_rng('zdc'), z_mean.shape)
        return z_mean + jnp.exp(0.5 * z_log_var) * epsilon
