import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange

from zdc.layers import Concatenate, ConvNeXtV2Block, ConvNeXtV2Embedding, ConvNeXtV2Stage, GlobalAveragePooling, Reshape, UpSample
from zdc.models.gan.gan import eval_fn, train_fn
from zdc.utils.data import load, batches
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, save_model, print_model, get_layers


class Discriminator(nn.Module):
    kernel_size: int = 3
    max_drop_rate: float = 0.33
    depths: tuple = (1, 1, 3, 1)
    projection_dims: tuple = (24, 48, 96, 192)
    drop_rates = [r.tolist() for r in jnp.split(jnp.linspace(0., max_drop_rate, sum(depths)), jnp.cumsum(jnp.array(depths))[:-1])]

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = ConvNeXtV2Embedding(patch_size=2, projection_dim=self.projection_dims[0])(img)

        for i, (projection_dim, drop_rates) in enumerate(zip(self.projection_dims, self.drop_rates)):
            patch_size = 2 if i > 0 else 1
            x = ConvNeXtV2Stage(patch_size, projection_dim, self.kernel_size, drop_rates)(x, training=training)

        x = GlobalAveragePooling()(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = Concatenate()(x, cond)
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x


class Generator(nn.Module):
    kernel_size: int = 3
    decoder_dim: int = 128

    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate()(z, cond)
        x = nn.Dense(6 * 6 * self.decoder_dim)(x)
        x = Reshape((-1, 6, 6, self.decoder_dim))(x)

        for _ in range(3):
            x = UpSample()(x)
            x = ConvNeXtV2Block(self.decoder_dim, self.kernel_size)(x, training=training)

        x = nn.Conv(1, kernel_size=(5, 5), padding='valid')(x)
        x = nn.relu(x)
        return x


class ConvNeXtGAN(nn.Module):
    def setup(self):
        self.discriminator = Discriminator()
        self.generator = Generator()

    def __call__(self, img, cond, rand_cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (img.shape[0], 32))
        generated = self.generator(z, rand_cond, training=training)
        real_output = self.discriminator(img, cond, training=training)
        fake_output = self.discriminator(generated, rand_cond, training=training)
        return generated, real_output, fake_output


class ConvNeXtGANGen(nn.Module):
    def setup(self):
        self.generator = Generator()

    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 32))
        return self.generator(z, cond, training=False)


if __name__ == '__main__':
    batch_size = 128
    n_reps = 5
    lr = 1e-5
    epochs = 200
    seed = 42

    key = jax.random.PRNGKey(seed)
    data_key, init_key, train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 7)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    f_train, f_val, f_test = tuple(map(lambda x: jax.random.permutation(*x), zip(jax.random.split(data_key, 3), (p_train, p_val, p_test))))
    r_sample, p_sample, f_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train, f_train))

    model, model_gen = ConvNeXtGAN(), ConvNeXtGANGen()
    params, state = init(model, init_key, r_sample, p_sample, f_sample)
    print_model(params)

    disc_optimizer = optax.adam(lr)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_optimizer = optax.adam(lr)
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(train_fn, model=model, disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer))
    eval_fn = jax.jit(partial(eval_fn, model=model, n_reps=n_reps))
    eval_metrics = ('disc_loss', 'gen_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc', 'mse', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='gan_convnext')
    os.makedirs('checkpoints/gan_convnext', exist_ok=True)

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

        save_model(params, state, f'checkpoints/gan_convnext/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, f_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log()
