from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, ConvNeXtV2Block, ConvNeXtV2Embedding, ConvNeXtV2Stage, GlobalAveragePooling, Reshape, UpSample
from zdc.models.gan.gan import eval_fn, train_fn
from zdc.utils.data import get_samples, load
from zdc.utils.nn import init, forward, get_layers, opt_with_cosine_schedule
from zdc.utils.train import train_loop


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
        x = nn.LayerNorm()(x)
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
        x = Reshape((6, 6, self.decoder_dim))(x)

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
    key = jax.random.PRNGKey(42)
    data_key, init_key, train_key = jax.random.split(key, 3)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    f_train, f_val, f_test = tuple(map(lambda x: jax.random.permutation(*x), zip(jax.random.split(data_key, 3), (p_train, p_val, p_test))))
    r_sample, p_sample, f_sample = get_samples(r_train, p_train, f_train)

    model, model_gen = ConvNeXtGAN(), ConvNeXtGANGen()
    params, state = init(model, init_key, r_sample, p_sample, f_sample, print_summary=True)

    disc_optimizer = opt_with_cosine_schedule(optax.adam, 1e-4)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_optimizer = opt_with_cosine_schedule(optax.adam, 1e-4)
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(train_fn, model=model, disc_optimizer=disc_optimizer, gen_optimizer=gen_optimizer))
    eval_fn = jax.jit(partial(eval_fn, model=model))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('disc_loss', 'gen_loss')
    eval_metrics = ('disc_loss', 'gen_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc', 'mse', 'mae', 'wasserstein')

    train_loop(
        'convnext_gan', train_fn, eval_fn, plot_fn, (r_train, p_train, f_train), (r_val, p_val, f_val), (r_test, p_test, f_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, (disc_opt_state, gen_opt_state), train_key, epochs=100, batch_size=128
    )
