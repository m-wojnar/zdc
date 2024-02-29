import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange

from zdc.layers import Concatenate, ConvBlock, Flatten, Reshape, Sampling, UpSample
from zdc.utils.data import load, batches
from zdc.utils.losses import kl_loss, mse_loss, mae_loss, wasserstein_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model
from zdc.utils.wasserstein import sum_channels_parallel


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img, cond, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = Flatten()(x)
        x = Concatenate()(x, cond)
        x = nn.Dense(20)(x)
        x = nn.relu(x)
        z_mean = nn.Dense(10)(x)
        z_log_var = nn.Dense(10)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


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
    return kl_weight * kl + mse, (state, kl, mse)


def eval_fn(params, state, key, img, cond, model, kl_weight, n_reps):
    def _eval_fn(subkey):
        (reconstructed, z_mean, z_log_var), _ = forward(model, params, state, subkey, img, cond, False)
        ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
        kl = kl_loss(z_mean, z_log_var)
        mse = mse_loss(img, reconstructed)
        mae = mae_loss(ch_true, ch_pred) / 5
        wasserstein = wasserstein_loss(ch_true, ch_pred)
        return kl_weight * kl + mse, kl, mse, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


if __name__ == '__main__':
    batch_size = 128
    kl_weight = 0.7
    n_reps = 5
    lr = 1e-4
    epochs = 100
    seed = 42

    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key, test_key, plot_key = jax.random.split(key, 5)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model, model_gen = VAE(), VAEGen()
    params, state = init(model, init_key, r_sample, p_sample)

    optimizer = optax.rmsprop(lr)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=kl_weight)))
    eval_fn = jax.jit(partial(eval_fn, model=model, kl_weight=kl_weight, n_reps=n_reps))
    eval_metrics = ('loss', 'kl', 'mse', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='vae')
    os.makedirs('checkpoints/vae', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        for batch in batches(r_train, p_train, batch_size=batch_size):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, loss, (state, kl, mse) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add({'loss': loss, 'kl': kl, 'mse': mse}, 'train')

        metrics.log(epoch)

        for batch in batches(r_val, p_val, batch_size=batch_size):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, forward(model_gen, params, state, subkey, p_sample)[0], epoch)

        save_model(params, state, f'checkpoints/vae/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
