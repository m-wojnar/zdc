from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.layers import Concatenate, ConvBlock, Flatten, Reshape, Sampling, UpSample
from zdc.utils.data import load
from zdc.utils.losses import kl_loss, mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    latent_dim: int = 10

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = Flatten()(x)
        x = Concatenate()(x, cond)
        x = nn.Dense(2 * self.latent_dim)(x)
        x = nn.relu(x)
        z_mean = nn.Dense(self.latent_dim)(x)
        z_log_var = nn.Dense(self.latent_dim)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate()(z, cond)
        x = nn.Dense(128 * 6 * 6)(x)
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


class VAE(nn.Module):
    @nn.compact
    def __call__(self, img, cond, training=True):
        z_mean, z_log_var, z = Encoder()(img, cond, training=training)
        reconstructed = Decoder()(z, cond, training=training)
        return reconstructed, z_mean, z_log_var


class VAEGen(nn.Module):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return Decoder()(z, cond, training=False)


def loss_fn(params, state, key, img, cond, model, kl_weight):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img, cond)
    kl = kl_loss(z_mean, z_log_var)
    mse = mse_loss(img, reconstructed)
    loss = kl_weight * kl + mse
    return loss, (state, loss, kl, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = VAE(), VAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'variational', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
