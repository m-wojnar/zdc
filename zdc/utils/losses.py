import jax.numpy as jnp
import optax

from zdc.utils.wasserstein import wasserstein_channels


def kl_loss(z_mean, z_log_var):
    return -0.5 * (1. + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var)).sum(axis=1).mean()


def mse_loss(x, y):
    return jnp.square(x - y).sum(axis=(1, 2)).mean()


def mae_loss(x, y):
    return jnp.abs(x - y).sum(axis=(1, 2)).mean()


def wasserstein_loss(x, generated):
    return wasserstein_channels(x, generated).mean()


def xentropy_loss(x, y):
    return optax.sigmoid_binary_cross_entropy(x, y).mean()
