from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, DenseBlock, Flatten, Reshape, UpSample
from zdc.utils.data import load
from zdc.utils.losses import xentropy_loss
from zdc.utils.nn import init, forward, gradient_step, get_layers, opt_with_cosine_schedule
from zdc.utils.train import train_loop, default_generate_fn


disc_optimizer = optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adam, b1=0.97, b2=0.96, eps=3.7e-5),
    peak_value=1.4e-5,
    pct_start=0.43,
    div_factor=41,
    final_div_factor=1700,
    epochs=100,
    batch_size=256
)

gen_optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adam, b1=0.76, b2=0.54, eps=1.8e-2, nesterov=True),
    peak_value=1e-4,
    pct_start=0.34,
    div_factor=260,
    final_div_factor=9500,
    epochs=100,
    batch_size=256
)


class ConvBlock(nn.Module):
    features: int
    kernel_size: int = 3
    strides: int = 1
    padding: str = 'same'
    use_bn: bool = False
    dropout_rate: float = None
    negative_slope: float = None
    max_pool_size: int = None

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(self.features, kernel_size=(self.kernel_size, self.kernel_size), strides=(self.strides, self.strides), padding=self.padding)(x)

        if self.use_bn:
            x = nn.BatchNorm()(x, use_running_average=not training)
        if self.dropout_rate is not None:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)
        if self.negative_slope is not None:
            x = nn.leaky_relu(x, negative_slope=self.negative_slope)
        if self.max_pool_size is not None:
            pool_size = (self.max_pool_size, self.max_pool_size)
            x = nn.max_pool(x, window_shape=pool_size, strides=pool_size)

        return x


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
        x = Reshape((13, 13, 128))(x)
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

    def gen(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return self.generator(z, cond, training=False)


def disc_loss_fn(disc_params, gen_params, state, forward_key, img, cond, rand_cond, model):
    (_, real_output, fake_output), state = forward(model, disc_params | gen_params, state, forward_key, img, cond, rand_cond)

    real_loss = xentropy_loss(real_output, jnp.ones_like(real_output))
    fake_loss = xentropy_loss(fake_output, jnp.zeros_like(fake_output))
    loss = real_loss + fake_loss

    disc_real_acc = (real_output > 0).mean()
    disc_fake_acc = (fake_output < 0).mean()
    return loss, (state, loss, disc_real_acc, disc_fake_acc)


def gen_loss_fn(gen_params, disc_params, state, forward_key, img, cond, rand_cond, model):
    (_, _, fake_output), state = forward(model, gen_params | disc_params, state, forward_key, img, cond, rand_cond)

    loss = xentropy_loss(fake_output, jnp.ones_like(fake_output))

    gen_acc = (fake_output > 0).mean()
    return loss, (state, loss, gen_acc)


def step_fn(params, carry, opt_state, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    state, key, img, cond = carry
    disc_opt_state, gen_opt_state = opt_state
    forward_key, data_key = jax.random.split(key)

    disc_params, gen_params = get_layers(params, 'discriminator'), get_layers(params, 'generator')
    rand_cond = jax.random.permutation(data_key, cond)

    disc_params_new, disc_opt_state, (_, disc_loss, disc_real_acc, disc_fake_acc) = gradient_step(
        disc_params, (gen_params, state, forward_key, img, cond, rand_cond), disc_opt_state, disc_optimizer, disc_loss_fn)
    gen_params_new, gen_opt_state, (state, gen_loss, gen_acc) = gradient_step(
        gen_params, (disc_params, state, forward_key, img, cond, rand_cond), gen_opt_state, gen_optimizer, gen_loss_fn)

    return disc_params_new | gen_params_new, (disc_opt_state, gen_opt_state), (state, disc_loss, gen_loss, disc_real_acc, disc_fake_acc, gen_acc)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = GAN()
    params, state = init(model, init_key, r_train[:5], p_train[:5], p_train[:5], print_summary=True)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, model=model),
        gen_loss_fn=partial(gen_loss_fn, model=model)
    ))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('disc_loss', 'gen_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc')

    train_loop(
        'gan', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key
    )
