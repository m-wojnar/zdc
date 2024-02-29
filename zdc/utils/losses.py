import jax.numpy as jnp


def kl_loss(z_mean, z_log_var):
    return -0.5 * (1. + z_log_var - jnp.square(z_mean) - jnp.exp(z_log_var)).sum(axis=1).mean()


def reconstruction_loss(x, reconstructed):
    return jnp.square(x - reconstructed).sum(axis=(1, 2)).mean()


def mae_loss(x, reconstructed):
    return jnp.abs(x - reconstructed).sum(axis=(1, 2)).mean()
