import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
import wandb
from flax import linen as nn
from tqdm import trange, tqdm

from zdc.layers import Concatenate, Sampling, Reshape, UpSample
from zdc.utils.data import load, batches
from zdc.utils.losses import kl_loss, reconstruction_loss, mae_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model
from zdc.utils.wasserstein import wasserstein_loss


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img, cond, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)

        x = nn.leaky_relu(x, negative_slope=0.1)
        x = Reshape((x.shape[0], -1))(x)
        x = Concatenate(axis=-1)(x, cond)
        x = nn.Dense(20)(x)
        x = nn.relu(x)

        z_mean = nn.Dense(10)(x)
        z_log_var = nn.Dense(10)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, cond, training=True):
        x = jnp.concatenate([z, cond], axis=-1)
        x = nn.Dense(128 * 6 * 6)(x)
        x = Reshape((-1, 6, 6, 128))(x)

        x = UpSample(2)(x)
        x = nn.Conv(128, kernel_size=(4, 4))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        x = UpSample(2)(x)
        x = nn.Conv(64, kernel_size=(4, 4))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

        x = UpSample(2)(x)
        x = nn.Conv(32, kernel_size=(4, 4))(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        x = nn.relu(x)

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
    def __call__(self, z, cond):
        return Decoder()(z, cond, training=False)


def loss_fn(params, state, key, img, cond, model, kl_weight):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img, cond)
    kl = kl_loss(z_mean, z_log_var)
    mse = reconstruction_loss(img, reconstructed)
    return mse + kl_weight * kl, (state, kl, mse)


def eval_fn(params, state, key, img, cond, model, kl_weight):
    (reconstructed, z_mean, z_log_var), _ = forward(model, params, state, key, img, cond, False)

    kl = kl_loss(z_mean, z_log_var)
    mse = reconstruction_loss(img, reconstructed)
    mae = mae_loss(img, reconstructed)
    wasserstein = wasserstein_loss(img, reconstructed)
    loss = mse + kl_weight * kl

    return loss, kl, mse, mae, wasserstein


if __name__ == '__main__':
    r_train, r_val, p_train, p_val = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model = VAE()
    key = jax.random.PRNGKey(42)
    params, state = init(model, key, r_sample, p_sample)

    optimizer = optax.rmsprop(1e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    eval_fn = jax.jit(partial(eval_fn, model=model, kl_weight=0.7))

    batch_size = 128
    kl_weight = 0.7
    epochs = 100

    wandb.init(project='zdc', job_type='train', name='vae')
    n_train, n_val = (r_train.shape[0] + batch_size - 1) // batch_size, (r_val.shape[0] + batch_size - 1) // batch_size

    model_gen = VAEGen()
    metrics = Metrics()
    best_loss = float('inf')
    os.makedirs('checkpoints/vae', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        metrics.reset()

        for r_batch, p_patch in tqdm(batches(r_train, p_train, batch_size=batch_size), desc='Train', leave=False, total=n_train):
            key, subkey = jax.random.split(key)
            params, opt_state, loss, (state, kl, mse) = train_fn(params, (state, subkey, r_batch, p_patch), opt_state)
            metrics.add({'loss_train': loss, 'kl_train': kl, 'mse_train': mse})

        metrics.collect()
        metrics.log(epoch)
        metrics.reset()

        for r_batch, p_patch in tqdm(batches(r_val, p_val, batch_size=batch_size), desc='Val', leave=False, total=n_val):
            key, subkey = jax.random.split(key)
            loss, kl, mse, mae, wasserstein = eval_fn(params, state, subkey, r_batch, p_patch)
            metrics.add({'loss_val': loss, 'kl_val': kl, 'mse_val': mse, 'mae_val': mae, 'wasserstein_val': wasserstein})

        metrics.collect()
        metrics.log(epoch)

        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (10, 10))
        metrics.plot_responses(r_sample, forward(model_gen, params, state, key, z, p_sample)[0])

        if metrics.metrics['loss_val'] < best_loss:
            best_loss = metrics.metrics['loss_val']
            save_model(params, state, f'checkpoints/vae/epoch_{epoch + 1}.pkl.lz4')
