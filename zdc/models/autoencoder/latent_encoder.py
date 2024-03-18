from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Flatten, Concatenate, DenseBlock
from zdc.models.autoencoder.variational import Decoder
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class LatentEncoder(nn.Module):
    latent_dim: int = 20
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, z, cond, training=True):
        x1 = DenseBlock(2 * self.hidden_dim, negative_slope=0.2)(z)
        x2 = DenseBlock(self.hidden_dim, negative_slope=0.2)(cond)
        x = Concatenate()(x1, x2)
        x = DenseBlock(4 * self.hidden_dim, negative_slope=0.2)(x)
        x = DenseBlock(4 * self.hidden_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim, negative_slope=0.2)(x)
        return x


class Encoder(nn.Module):
    latent_dim: int = 20

    @nn.compact
    def __call__(self, img, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = Flatten()(x)
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)
        return x


class LEVAE(nn.Module):
    noise_dim: int = 10

    @nn.compact
    def __call__(self, img, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.noise_dim))
        none = jnp.zeros((cond.shape[0], 0))

        enc = Encoder()(img, training=training)
        le = LatentEncoder()(z, cond, training=training)
        reconstructed = Decoder()(none, enc, training=training)
        return reconstructed, enc, le


class LEVAEGen(nn.Module):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        none = jnp.zeros((cond.shape[0], 0))

        le = LatentEncoder()(z, cond, training=False)
        return Decoder()(none, le, training=False)


def loss_fn(params, state, key, img, cond, model, enc_weight):
    (reconstructed, enc, le), state = forward(model, params, state, key, img, cond)
    mse_enc = mse_loss(enc, le)
    mse_rec = mse_loss(img, reconstructed)
    loss = enc_weight * mse_enc + mse_rec
    return loss, (state, loss, mse_enc, mse_rec)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = LEVAE(), LEVAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, enc_weight=10.)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'mse_enc', 'mse_rec')

    train_loop(
        'latent_encoder', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
