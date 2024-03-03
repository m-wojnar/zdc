import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange

from zdc.layers import Concatenate, ConvBlock, Flatten, Reshape, UpSample
from zdc.models import PARTICLE_SHAPE
from zdc.utils.data import load, batches
from zdc.utils.losses import mse_loss, mae_loss, wasserstein_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model, print_model
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
    return cond_weight * mse_cond + mse_rec, (state, mse_cond, mse_rec)


def eval_fn(params, state, key, img, cond, model, cond_weight, n_reps):
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
    batch_size = 128
    cond_weight = 1.0
    n_reps = 5
    lr = 1e-4
    epochs = 100
    seed = 42

    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key, test_key, plot_key = jax.random.split(key, 5)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model, model_gen = SupervisedAE(), SupervisedAEGen()
    params, state = init(model, init_key, r_sample)
    print_model(params)

    optimizer = optax.rmsprop(lr)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=cond_weight)))
    eval_fn = jax.jit(partial(eval_fn, model=model, cond_weight=cond_weight, n_reps=n_reps))
    eval_metrics = ('loss', 'mse_cond', 'mse_rec', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='supervised_ae')
    os.makedirs('checkpoints/supervised_ae', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        for batch in batches(r_train, p_train, batch_size=batch_size):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, loss, (state, mse_cond, mse_rec) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add({'loss': loss, 'mse_cond': mse_cond, 'mse_rec': mse_rec}, 'train')

        metrics.log(epoch)

        for batch in batches(r_val, p_val, batch_size=batch_size):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, forward(model_gen, params, state, subkey, p_sample)[0], epoch)

        save_model(params, state, f'checkpoints/supervised_ae/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
