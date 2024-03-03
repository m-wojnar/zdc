import jax.numpy as jnp
import optax

from zdc.utils.wasserstein import wasserstein_channels


def kl_loss(mean, log_var):
    return -0.5 * (1. + log_var - jnp.square(mean) - jnp.exp(log_var)).sum(axis=1).mean()


def mse_loss(x, y):
    return jnp.square(x - y).reshape(x.shape[0], -1).sum(axis=-1).mean()


def mae_loss(x, y):
    return jnp.abs(x - y).reshape(x.shape[0], -1).sum(axis=-1).mean()


def wasserstein_loss(ch_true, ch_pred):
    return wasserstein_channels(ch_true, ch_pred).mean()


def xentropy_loss(x, y):
    return optax.sigmoid_binary_cross_entropy(x, y).reshape(x.shape[0], -1).sum(axis=-1).mean()
