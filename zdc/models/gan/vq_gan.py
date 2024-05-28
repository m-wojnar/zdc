from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.architectures.vit import Encoder, Decoder
from zdc.layers import Flatten
from zdc.models.quantization.vq_vae import VQVAE
from zdc.utils.data import load
from zdc.utils.losses import mae_loss, mse_loss, perceptual_loss, xentropy_loss
from zdc.utils.nn import init, forward, get_layers, opt_with_cosine_schedule, gradient_step
from zdc.utils.train import train_loop


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


class Discriminator(nn.Module):
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = Encoder(self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)(img, cond, training=training)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        x = nn.gelu(x)
        x = nn.Dense(1)(x)
        x = Flatten()(x)
        return x


class VQGAN(nn.Module):
    vq_encoder_type: nn.Module
    vq_decoder_type: nn.Module
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1
    latent_dim: int = 10

    def setup(self):
        self.discriminator = Discriminator(self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)
        self.generator = VQVAE(self.vq_encoder_type, self.vq_decoder_type)

    def __call__(self, img, cond, rand_img, rand_cond, training=True):
        reconstructed, encoded, discrete, quantized = self.generator(rand_img, training=training)
        real_output = self.discriminator(img, cond, training=training)
        fake_output = self.discriminator(reconstructed, rand_cond, training=training)
        return reconstructed, encoded, discrete, quantized, real_output, fake_output

    def reconstruct(self, img):
        return self.generator(img, training=False)[0]

    def gen(self, discrete):
        return self.generator.gen(discrete)


def disc_loss_fn(disc_params, gen_params, state, forward_key, step, *x, model):
    (*_, real_output, fake_output), state = forward(model, disc_params | gen_params, state, forward_key, *x)

    disc_real_acc = (real_output > 0).mean()
    disc_fake_acc = (fake_output < 0).mean()
    adv_weight = jax.lax.cond(step < 10000, lambda _: 0., lambda _: 1., None)

    real_loss = xentropy_loss(real_output, jnp.ones_like(real_output))
    fake_loss = xentropy_loss(fake_output, jnp.zeros_like(fake_output))
    loss = adv_weight * (real_loss + fake_loss)

    return loss, (state, loss, disc_real_acc, disc_fake_acc)


def gen_loss_fn(gen_params, disc_params, state, forward_key, step, *x, model, perceptual_loss_fn, loss_weights, commitment_cost=0.25):
    (generated, encoded, discrete, quantized, _, fake_output), state = forward(model, gen_params | disc_params, state, forward_key, *x)
    img, *_ = x

    e_loss = mse_loss(jax.lax.stop_gradient(quantized), encoded)
    q_loss = mse_loss(quantized, jax.lax.stop_gradient(encoded))
    vq_loss = commitment_cost * e_loss + q_loss

    l1_loss = mae_loss(img, generated)
    l2_loss = mse_loss(img, generated)
    perc_loss = perceptual_loss_fn(img, generated)
    adv_loss = xentropy_loss(fake_output, jnp.ones_like(fake_output))

    gen_acc = (fake_output > 0).mean()
    avg_prob = jnp.mean(discrete, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_prob * jnp.log(avg_prob + 1e-10)))

    l1_weight, l2_weight, perc_weight, adv_weight = loss_weights
    adv_weight = jax.lax.cond(step < 10000, lambda _: 0., lambda _: adv_weight, None)
    loss = vq_loss + l1_weight * l1_loss + l2_weight * l2_loss + perc_weight * perc_loss + adv_weight * adv_loss

    return loss, (state, loss, vq_loss, l1_loss, l2_loss, perc_loss, adv_loss, gen_acc, perplexity)


def step_fn(params, carry, opt_state, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    state, key, img, cond = carry
    disc_opt_state, gen_opt_state, step = opt_state
    forward_key, data_key = jax.random.split(key)

    disc_params, gen_params = get_layers(params, 'discriminator'), get_layers(params, 'generator')
    permutation = jax.random.permutation(data_key, img.shape[0])
    rand_img, rand_cond = img[permutation], cond[permutation]

    disc_params_new, disc_opt_state, (_, *disc_losses) = gradient_step(
        disc_params, (gen_params, state, forward_key, step, img, cond, rand_img, rand_cond), disc_opt_state, disc_optimizer, disc_loss_fn)
    gen_params_new, gen_opt_state, (state, *gen_losses) = gradient_step(
        gen_params, (disc_params, state, forward_key, step, img, cond, rand_img, rand_cond), gen_opt_state, gen_optimizer, gen_loss_fn)

    return disc_params_new | gen_params_new, (disc_opt_state, gen_opt_state, step + 1), (state, *disc_losses, *gen_losses)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VQGAN(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5], r_train[:5], p_train[:5], print_summary=True)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, model=model),
        gen_loss_fn=partial(gen_loss_fn, model=model, perceptual_loss_fn=perceptual_loss(), loss_weights=(0.1 / (44 ** 2), 1.0 / (44 ** 2), 0.1, 0.1 / (6 ** 2)))
    ))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0], method='reconstruct')[0])
    train_metrics = ('disc_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_loss', 'vq_loss', 'l1_loss', 'l2_loss', 'perc_loss', 'adv_loss', 'gen_acc', 'perplexity')

    train_loop(
        'vq_gan', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state, 0), train_key
    )
