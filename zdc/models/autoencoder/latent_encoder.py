from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import MLP, Flatten, Concatenate
from zdc.models.autoencoder.variational import Decoder
from zdc.utils.data import get_samples, load
from zdc.utils.losses import mse_loss, mae_loss, wasserstein_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop
from zdc.utils.wasserstein import sum_channels_parallel


class LatentEncoder(nn.Module):
    latent_dim: int = 10

    @nn.compact
    def __call__(self, cond, training=True):
        return MLP([64, 64, self.latent_dim])(cond)


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
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.latent_dim)(x)
        return x


class LEVAE(nn.Module):
    @nn.compact
    def __call__(self, img, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        enc = Encoder()(img, cond, training=training)
        le = LatentEncoder()(cond, training=training)
        reconstructed = Decoder()(z, enc, training=training)
        return reconstructed, enc, le


class LEVAEGen(nn.Module):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        le = LatentEncoder()(cond, training=False)
        return Decoder()(z, le, training=False)


def loss_fn(params, state, key, img, cond, model, enc_weight):
    (reconstructed, enc, le), state = forward(model, params, state, key, img, cond)
    mse_enc = mse_loss(enc, le)
    mse_rec = mse_loss(img, reconstructed)
    loss = enc_weight * mse_enc + mse_rec
    return loss, (state, loss, mse_enc, mse_rec)


def eval_fn(params, state, key, img, cond, model, enc_weight, n_reps=5):
    def _eval_fn(subkey):
        (reconstructed, enc, le), _ = forward(model, params, state, subkey, img, cond, False)
        ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
        mse_enc = mse_loss(enc, le)
        mse_rec = mse_loss(img, reconstructed)
        mae = mae_loss(ch_true, ch_pred) / 5
        wasserstein = wasserstein_loss(ch_true, ch_pred)
        return enc_weight * mse_enc + mse_rec, mse_enc, mse_rec, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)

    model, model_gen = LEVAE(), LEVAEGen()
    params, state = init(model, init_key, r_sample, p_sample, print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, enc_weight=10.)))
    eval_fn = jax.jit(partial(eval_fn, model=model, enc_weight=10.))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('loss', 'mse_enc', 'mse_rec')
    eval_metrics = ('loss', 'mse_enc', 'mse_rec', 'mae', 'wasserstein')

    train_loop(
        'latent_encoder', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
