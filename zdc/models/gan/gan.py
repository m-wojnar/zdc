import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange

from zdc.layers import Concatenate, ConvBlock, DenseBlock, Flatten, Reshape, UpSample
from zdc.utils.data import load, batches
from zdc.utils.losses import mse_loss, mae_loss, wasserstein_loss, xentropy_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model, get_layers, print_model
from zdc.utils.wasserstein import sum_channels_parallel


class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, img, cond, training=True):
        x = ConvBlock(32, kernel_size=3, padding='valid', use_bn=True, dropout_rate=0.2, negative_slope=0.1, max_pool_size=2)(img, training=training)
        x = ConvBlock(16, kernel_size=3, padding='valid', use_bn=True, dropout_rate=0.2, negative_slope=0.1, max_pool_size=2)(x, training=training)
        x = Flatten()(x)
        x = Concatenate()(x, cond)
        x = DenseBlock(128, use_bn=True, dropout_rate=0.2, negative_slope=0.1)(x, training=training)
        x = DenseBlock(64, use_bn=True, dropout_rate=0.2, negative_slope=0.1)(x, training=training)
        x = nn.Dense(1)(x)
        return x


class Generator(nn.Module):
    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate()(z, cond)
        x = DenseBlock(128 * 2, use_bn=True, dropout_rate=0.2, negative_slope=0.1)(x, training=training)
        x = DenseBlock(128 * 13 * 13, use_bn=True, dropout_rate=0.2, negative_slope=0.1)(x, training=training)
        x = Reshape((-1, 13, 13, 128))(x)
        x = UpSample()(x)
        x = ConvBlock(128, kernel_size=3, padding='valid', use_bn=True, dropout_rate=0.2, negative_slope=0.1)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(64, kernel_size=3, padding='valid', use_bn=True, dropout_rate=0.2, negative_slope=0.1)(x, training=training)
        x = ConvBlock(1, kernel_size=3, padding='valid', negative_slope=0.)(x, training=training)
        return x


class GAN(nn.Module):
    def setup(self):
        self.discriminator = Discriminator()
        self.generator = Generator()

    def __call__(self, img, cond, rand_cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (img.shape[0], 10))
        generated = self.generator(z, rand_cond, training=training)
        real_output = self.discriminator(img, cond, training=training)
        fake_output = self.discriminator(generated, rand_cond, training=training)
        return generated, real_output, fake_output


class GANGen(nn.Module):
    def setup(self):
        self.generator = Generator()

    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return self.generator(z, cond, training=False)


def disc_loss_fn(real_output, fake_output):
    real_loss = xentropy_loss(real_output, jnp.ones_like(real_output))
    fake_loss = xentropy_loss(fake_output, jnp.zeros_like(fake_output))
    return real_loss + fake_loss


def gen_loss_fn(fake_output):
    return xentropy_loss(fake_output, jnp.ones_like(fake_output))


def train_fn(params, state, key, img, cond, rand_cond, disc_opt_state, gen_opt_state, model, disc_optimizer, gen_optimizer):
    def _disc_loss_fn(params, other_params, state):
        (_, real_output, fake_output), _ = forward(model, params | other_params, state, key, img, cond, rand_cond)
        return disc_loss_fn(real_output, fake_output), None

    def _gen_loss_fn(params, other_params, state):
        (_, _, fake_output), state = forward(model, params | other_params, state, key, img, cond, rand_cond)
        return gen_loss_fn(fake_output), state

    disc_params, disc_opt_state, disc_loss, _ = gradient_step(
        get_layers(params, 'discriminator'), (get_layers(params, 'generator'), state), disc_opt_state, disc_optimizer, _disc_loss_fn)
    gen_params, gen_opt_state, gen_loss, state = gradient_step(
        get_layers(params, 'generator'), (get_layers(params, 'discriminator'), state), gen_opt_state, gen_optimizer, _gen_loss_fn)

    return disc_params | gen_params, state, disc_opt_state, gen_opt_state, disc_loss, gen_loss


def eval_fn(params, state, key, img, cond, rand_cond, model, n_reps):
    def _eval_fn(subkey):
        (generated, real_output, fake_output), _ = forward(model, params, state, subkey, img, cond, rand_cond, False)
        ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(generated)
        disc_loss = disc_loss_fn(real_output, fake_output)
        gen_loss = gen_loss_fn(fake_output)
        disc_real_acc = (real_output > 0).mean()
        disc_fake_acc = (fake_output < 0).mean()
        gen_acc = (fake_output > 0).mean()
        mse = mse_loss(img, generated)
        mae = mae_loss(ch_true, ch_pred) / 5
        wasserstein = wasserstein_loss(ch_true, ch_pred)
        return disc_loss, gen_loss, disc_real_acc, disc_fake_acc, gen_acc, mse, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


if __name__ == '__main__':
    batch_size = 128
    n_reps = 5
    lr = 1e-4
    epochs = 200
    seed = 42

    key = jax.random.PRNGKey(seed)
    data_key, init_key, train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 7)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    f_train, f_val, f_test = tuple(map(lambda x: jax.random.permutation(*x), zip(jax.random.split(data_key, 3), (p_train, p_val, p_test))))
    r_sample, p_sample, f_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train, f_train))

    model, model_gen = GAN(), GANGen()
    params, state = init(model, init_key, r_sample, p_sample, f_sample)
    print_model(params)

    disc_optimizer = optax.adam(lr)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_optimizer = optax.adam(lr)
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(train_fn, model=model, disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer))
    eval_fn = jax.jit(partial(eval_fn, model=model, n_reps=n_reps))
    eval_metrics = ('disc_loss', 'gen_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc', 'mse', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='gan')
    os.makedirs('checkpoints/gan', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(r_train, p_train, f_train, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, state, disc_opt_state, gen_opt_state, disc_loss, gen_loss = train_fn(params, state, subkey, *batch, disc_opt_state, gen_opt_state)
            metrics.add({'disc_loss': disc_loss, 'gen_loss': gen_loss}, 'train')

        metrics.log(epoch)

        for batch in batches(r_val, p_val, f_val, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, forward(model_gen, params, state, subkey, p_sample)[0], epoch)

        save_model(params, state, f'checkpoints/gan/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, f_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
