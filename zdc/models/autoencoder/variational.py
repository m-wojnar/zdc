import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange, tqdm

from zdc.layers import Concatenate, Flatten, Reshape, Sampling, UpSample
from zdc.utils.data import load, batches
from zdc.utils.losses import kl_loss, mse_loss, mae_loss
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
        x = Flatten()(x)
        x = Concatenate(axis=-1)(cond, x)
        x = nn.Dense(20)(x)
        x = nn.relu(x)

        z_mean = nn.Dense(10)(x)
        z_log_var = nn.Dense(10)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate(axis=-1)(z, cond)
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
    mse = mse_loss(img, reconstructed)
    return kl_weight * kl + mse, (state, kl, mse)


def eval_fn(params, state, key, img, cond, model, kl_weight, n_reps):
    def _eval_fn(subkey):
        (reconstructed, z_mean, z_log_var), _ = forward(model, params, state, subkey, img, cond, False)
        kl = kl_loss(z_mean, z_log_var)
        mse = mse_loss(img, reconstructed)
        mae = mae_loss(img, reconstructed)
        wasserstein = wasserstein_loss(img, reconstructed)
        return kl_weight * kl + mse, kl, mse, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


if __name__ == '__main__':
    batch_size = 128
    kl_weight = 0.7
    n_reps = 5
    lr = 1e-4
    epochs = 100
    max_patience = 10
    seed = 42

    r_train, r_val, p_train, p_val = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))
    n_train, n_val = (r_train.shape[0] + batch_size - 1) // batch_size, (r_val.shape[0] + batch_size - 1) // batch_size

    model = VAE()
    model_gen = VAEGen()

    key = jax.random.PRNGKey(seed)
    params, state = init(model, key, r_sample, p_sample)

    optimizer = optax.rmsprop(lr)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=kl_weight)))
    eval_fn = jax.jit(partial(eval_fn, model=model, kl_weight=kl_weight, n_reps=n_reps))

    metrics = Metrics(job_type='train', name='vae')
    best_loss, no_improvement_steps = float('inf'), 0
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
            no_improvement_steps = 0
        else:
            no_improvement_steps += 1

        if no_improvement_steps > max_patience:
            break
