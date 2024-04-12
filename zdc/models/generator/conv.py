from functools import partial

import jax
from flax import linen as nn

from zdc.models.autoencoder.variational import Decoder, optimizer
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


class Generator(nn.Module):
    decoder_type: nn.Module
    noise_dim: int = 4

    def setup(self):
        self.decoder = self.decoder_type()

    def __call__(self, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 6 * 6, self.noise_dim))
        return self.decoder(z, cond, training=training)

    def gen(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 6 * 6, self.noise_dim))
        return self.decoder(z, cond, training=False)


def loss_fn(params, state, key, img, cond, model):
    reconstructed, state = forward(model, params, state, key, cond)
    loss = mse_loss(img, reconstructed)
    return loss, (state, loss)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key,)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = Generator(Decoder)
    params, state = init(model, init_key, p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss',)

    train_loop(
        'decoder', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
