from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from zdc.architectures.vit import Encoder
from zdc.layers import Flatten
from zdc.models.gan.gan import Generator, disc_optimizer, gen_optimizer
from zdc.utils.data import load
from zdc.utils.losses import xentropy_loss
from zdc.utils.nn import init, forward, get_layers, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


class Discriminator(nn.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = Encoder(self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)(img, cond, training=training)
        x = nn.Dense(128)(x)
        e = Flatten()(x)
        x = nn.Dense(128)(e)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x, e


class SDIGAN(nn.Module):
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1
    latent_dim: int = 10

    def setup(self):
        self.discriminator = Discriminator(self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)
        self.generator = Generator(self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)

    def __call__(self, img, cond, rand_cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (img.shape[0], self.latent_dim))
        generated = self.generator(z, rand_cond, training=training)
        real_output, real_embed = self.discriminator(img, cond, training=training)
        fake_output, fake_embed = self.discriminator(generated, rand_cond, training=training)
        return generated, z, real_output, real_embed, fake_output, fake_embed

    def gen(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.latent_dim))
        return self.generator(z, cond, training=False)


def disc_loss_fn(disc_params, gen_params, state, forward_key, img, cond, rand_cond, model):
    (_, z, real_output, _, fake_output, _), state = forward(model, disc_params | gen_params, state, forward_key, img, cond, rand_cond)

    real_loss = xentropy_loss(real_output, jnp.ones_like(real_output))
    fake_loss = xentropy_loss(fake_output, jnp.zeros_like(fake_output))
    loss = real_loss + fake_loss

    disc_real_acc = (real_output > 0).mean()
    disc_fake_acc = (fake_output < 0).mean()
    return loss, (state, loss, disc_real_acc, disc_fake_acc)


def sdi_loss(diversity, z_1, z_2, embed_1, embed_2):
    dz = jnp.abs(z_1 - z_2).mean(axis=-1, keepdims=True)
    dG = jnp.abs(embed_1 - embed_2).mean(axis=-1, keepdims=True)
    return (diversity * dz / dG).mean()


def gen_loss_fn(gen_params, disc_params, state, forward_key, img, cond, rand_cond, diversity, model, div_weight=1.0):
    _, forward_key_2 = jax.random.split(forward_key)

    (_, z_1, _, _, fake_output, embed_1), state = forward(model, gen_params | disc_params, state, forward_key, img, cond, rand_cond)
    (_, z_2, _, _, _, embed_2), _ = forward(model, gen_params | disc_params, state, forward_key_2, img, cond, rand_cond)

    adv_loss = xentropy_loss(fake_output, jnp.ones_like(fake_output))
    div_loss = sdi_loss(diversity, z_1, z_2, embed_1, embed_2)
    loss = adv_loss + div_weight * div_loss

    gen_acc = (fake_output > 0).mean()
    return loss, (state, loss, adv_loss, div_loss, gen_acc)


def step_fn(params, carry, opt_state, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    state, key, img, cond, diversity = carry
    disc_opt_state, gen_opt_state = opt_state
    forward_key, data_key = jax.random.split(key)

    disc_params, gen_params = get_layers(params, 'discriminator'), get_layers(params, 'generator')
    rand_cond = jax.random.permutation(data_key, cond)

    disc_params_new, disc_opt_state, (_, disc_loss, disc_real_acc, disc_fake_acc) = gradient_step(
        disc_params, (gen_params, state, forward_key, img, cond, rand_cond), disc_opt_state, disc_optimizer, disc_loss_fn)
    gen_params_new, gen_opt_state, (state, gen_loss, _, div_loss, gen_acc) = gradient_step(
        gen_params, (disc_params, state, forward_key, img, cond, rand_cond, diversity), gen_opt_state, gen_optimizer, gen_loss_fn)

    return disc_params_new | gen_params_new, (disc_opt_state, gen_opt_state), (state, disc_loss, gen_loss, div_loss, disc_real_acc, disc_fake_acc, gen_acc)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()
    div_train = jnp.zeros(p_train.shape[0], dtype=float)

    for u in jnp.unique(p_train, axis=0):
        mask = jnp.all(p_train == u, axis=1)
        div_train = div_train.at[mask].set(jnp.std(r_train[mask], axis=0).sum())

    div_min, div_max = jnp.min(div_train), jnp.max(div_train)
    div_train = (div_train - div_min) / (div_max - div_min)
    div_train = div_train.reshape(-1, 1)

    model = SDIGAN()
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
    train_metrics = ('disc_loss', 'gen_loss', 'div_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc')

    train_loop(
        'sdi_gan', train_fn, None, generate_fn, (r_train, p_train, div_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key
    )
