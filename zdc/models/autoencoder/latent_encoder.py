import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange

from zdc.layers import MLP, Flatten, Concatenate
from zdc.models.autoencoder.variational import Decoder
from zdc.utils.data import load, batches
from zdc.utils.losses import kl_loss, mse_loss, mae_loss, wasserstein_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model, print_model
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
    return enc_weight * mse_enc + mse_rec, (state, mse_enc, mse_rec)


def eval_fn(params, state, key, img, cond, model, enc_weight, n_reps):
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
    batch_size = 128
    enc_weight = 10.0
    n_reps = 5
    lr = 1e-4
    epochs = 100
    seed = 42

    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 6)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model, model_gen = LEVAE(), LEVAEGen()
    params, state = init(model, init_key, r_sample, p_sample)
    print_model(params)

    optimizer = optax.rmsprop(lr)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, enc_weight=enc_weight)))
    eval_fn = jax.jit(partial(eval_fn, model=model, enc_weight=enc_weight, n_reps=n_reps))
    eval_metrics = ('loss', 'mse_enc', 'mse_rec', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='latent_encoder')
    os.makedirs('checkpoints/latent_encoder', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(r_train, p_train, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, loss, (state, mce_enc, mse_rec) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add({'loss': loss, 'mse_enc': mce_enc, 'mse_rec': mse_rec}, 'train')

        metrics.log(epoch)

        for batch in batches(r_val, p_val, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, forward(model_gen, params, state, subkey, p_sample)[0], epoch)

        save_model(params, state, f'checkpoints/latent_encoder/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
