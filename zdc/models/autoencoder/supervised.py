from functools import partial

import jax
from flax import linen as nn

from zdc.architectures.conv import Decoder, Encoder, optimizer
from zdc.layers import Flatten
from zdc.models import PARTICLE_SHAPE
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


class SupervisedAE(nn.Module):
    encoder_type: nn.Module
    decoder_type: nn.Module
    noise_dim: int = 4
    hidden_dim: int = 64

    def setup(self):
        self.encoder = self.encoder_type()
        self.flatten = Flatten()
        self.pre_latent = nn.Dense(*PARTICLE_SHAPE)
        self.decoder = self.decoder_type()

    def __call__(self, img, training=True):
        x = self.encoder(img, training=training)
        x = self.flatten(x)
        cond = self.pre_latent(x)

        z = jax.random.normal(self.make_rng('zdc'), (img.shape[0], 6 * 6, self.noise_dim))
        reconstructed = self.decoder(z, cond, training=training)
        return reconstructed, cond

    def gen(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 6 * 6, self.noise_dim))
        return self.decoder(z, cond, training=False)


def loss_fn(params, state, key, img, cond, model, cond_weight=1.0):
    (reconstructed, encoder_cond), state = forward(model, params, state, key, img)
    mse_cond = mse_loss(cond, encoder_cond)
    mse_rec = mse_loss(img, reconstructed)
    loss = cond_weight * mse_cond + mse_rec
    return loss, (state, loss, mse_cond, mse_rec)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = SupervisedAE(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss', 'mse_cond', 'mse_rec')

    train_loop(
        'supervised', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
