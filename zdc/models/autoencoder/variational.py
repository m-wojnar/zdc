from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from zdc.architectures.vit import Encoder, Decoder, optimizer
from zdc.layers import Flatten, Reshape, Concatenate
from zdc.utils.data import load
from zdc.utils.losses import kl_loss, mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


class Sampling(nn.Module):
    @nn.compact
    def __call__(self, z_mean, z_log_var):
        epsilon = jax.random.normal(self.make_rng('zdc'), z_mean.shape)
        return z_mean + jnp.exp(0.5 * z_log_var) * epsilon


class VAE(nn.Module):
    encoder_type: nn.Module
    decoder_type: nn.Module
    latent_dim: int = 10
    hidden_dim: int = 64

    def setup(self):
        self.encoder = self.encoder_type()
        self.pre_latent = nn.Dense(self.hidden_dim)
        self.flatten = Flatten()

        self.dense_shared = nn.Dense(2 * self.latent_dim)
        self.dense_mean = nn.Dense(self.latent_dim)
        self.dense_log_var = nn.Dense(self.latent_dim)
        self.sample = Sampling()

        self.concatenate = Concatenate()
        self.post_latent = nn.Dense(6 * 6 * self.hidden_dim)
        self.reshape = Reshape((6 * 6, self.hidden_dim))
        self.decoder = self.decoder_type()

    def __call__(self, img, cond, training=True):
        x = self.encoder(img, cond, training=training)
        x = self.pre_latent(x)
        x = self.flatten(x)

        x = self.dense_shared(x)
        x = nn.relu(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sample(z_mean, z_log_var)

        z = self.concatenate(z, cond)
        z = self.post_latent(z)
        z = self.reshape(z)
        reconstructed = self.decoder(z, training=training)
        return reconstructed, z_mean, z_log_var

    def gen(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.latent_dim))
        z = self.concatenate(z, cond)
        z = self.post_latent(z)
        z = self.reshape(z)
        return self.decoder(z, training=False)


def loss_fn(params, state, key, img, cond, model, kl_weight=0.7):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img, cond)
    kl = kl_loss(z_mean, z_log_var)
    mse = mse_loss(img, reconstructed)
    loss = kl_weight * kl + mse
    return loss, (state, loss, kl, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VAE(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'variational', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
