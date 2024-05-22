from functools import partial

import jax
from flax import linen as nn

from zdc.architectures.vit import Encoder, Decoder, optimizer
from zdc.layers import Concatenate, Flatten, Reshape
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


class NoiseGenerator(nn.Module):
    latent_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z, cond, training=True):
        x1 = nn.Dense(2 * self.hidden_dim)(z)
        x1 = nn.leaky_relu(x1, negative_slope=0.2)
        x2 = nn.Dense(self.hidden_dim)(cond)
        x2 = nn.leaky_relu(x2, negative_slope=0.2)
        x = Concatenate()(x1, x2)
        x = nn.Dense(4 * self.hidden_dim)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(4 * self.hidden_dim)(x)
        x = nn.leaky_relu(x, negative_slope=0.2)
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)
        return x


class AutoencoderNG(nn.Module):
    encoder_type: nn.Module
    decoder_type: nn.Module
    noise_dim: int = 10
    latent_dim: int = 20
    hidden_dim: int = 64
    hidden_dim_ng: int = 128

    def setup(self):
        self.noise_generator = NoiseGenerator(self.latent_dim, self.hidden_dim_ng)

        self.encoder = self.encoder_type()
        self.flatten = Flatten()
        self.pre_latent = nn.Dense(self.latent_dim)

        self.post_latent = nn.Dense(6 * 6 * self.hidden_dim)
        self.reshape = Reshape((6 * 6, self.hidden_dim))
        self.decoder = self.decoder_type()

    def __call__(self, img, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.noise_dim))
        ng = self.noise_generator(z, cond, training=training)

        x = self.encoder(img, training=training)
        x = self.flatten(x)
        x = self.pre_latent(x)
        enc = nn.relu(x)

        x = self.post_latent(enc)
        x = self.reshape(x)
        reconstructed = self.decoder(x, training=training)

        return reconstructed, enc, ng

    def gen(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.noise_dim))
        ng = self.noise_generator(z, cond, training=False)
        ng = self.post_latent(ng)
        ng = self.reshape(ng)
        return self.decoder(ng, training=False)


def loss_fn(params, state, key, img, cond, model, enc_weight=10.0):
    (reconstructed, enc, le), state = forward(model, params, state, key, img, cond)
    mse_enc = mse_loss(enc, le)
    mse_rec = mse_loss(img, reconstructed)
    loss = enc_weight * mse_enc + mse_rec
    return loss, (state, loss, mse_enc, mse_rec)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = AutoencoderNG(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss', 'mse_enc', 'mse_rec')

    train_loop(
        'noise_generator_mse', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
