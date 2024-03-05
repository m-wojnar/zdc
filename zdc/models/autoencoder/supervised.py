from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, ConvBlock, Flatten, Reshape, UpSample
from zdc.models import PARTICLE_SHAPE
from zdc.utils.data import load
from zdc.utils.losses import mse_loss, mae_loss, wasserstein_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop
from zdc.utils.wasserstein import sum_channels_parallel


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = Flatten()(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(*PARTICLE_SHAPE)(x)
        return x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate()(z, cond)
        x = nn.Dense(128 * 6 * 6)(x)
        x = Reshape((-1, 6, 6, 128))(x)
        x = UpSample()(x)
        x = ConvBlock(128, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(64, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(32, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = nn.Conv(1, kernel_size=(5, 5), padding='valid')(x)
        x = nn.relu(x)
        return x


class SupervisedAE(nn.Module):
    @nn.compact
    def __call__(self, img, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (img.shape[0], 10))
        cond = Encoder()(img, training=training)
        reconstructed = Decoder()(z, cond, training=training)
        return reconstructed, cond


class SupervisedAEGen(nn.Module):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return Decoder()(z, cond, training=False)


def loss_fn(params, state, key, img, cond, model, cond_weight):
    (reconstructed, encoder_cond), state = forward(model, params, state, key, img)
    mse_cond = mse_loss(cond, encoder_cond)
    mse_rec = mse_loss(img, reconstructed)
    loss = cond_weight * mse_cond + mse_rec
    return loss, (state, loss, mse_cond, mse_rec)


def eval_fn(params, state, key, img, cond, model, cond_weight, n_reps=5):
    def _eval_fn(subkey):
        (reconstructed, encoder_cond), _ = forward(model, params, state, subkey, img, False)
        ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
        mse_cond = mse_loss(cond, encoder_cond)
        mse_rec = mse_loss(img, reconstructed)
        mae = mae_loss(ch_true, ch_pred) / 5
        wasserstein = wasserstein_loss(ch_true, ch_pred)
        return cond_weight * mse_cond + mse_rec, mse_cond, mse_rec, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model, model_gen = SupervisedAE(), SupervisedAEGen()
    params, state = init(model, init_key, r_sample, print_summary=True)

    optimizer = optax.rmsprop(1e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=1.)))
    eval_fn = jax.jit(partial(eval_fn, model=model, cond_weight=1.))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('loss', 'mse_cond', 'mse_rec')
    eval_metrics = ('loss', 'mse_cond', 'mse_rec', 'mae', 'wasserstein')

    train_loop(
        'supervised', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
