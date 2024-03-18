from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.models.autoencoder.variational import Decoder as DecoderBlock
from zdc.utils.data import get_samples, load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return DecoderBlock()(z, cond, training=training)


def loss_fn(params, state, key, img, cond, model):
    reconstructed, state = forward(model, params, state, key, cond)
    loss = mse_loss(img, reconstructed)
    return loss, (state, loss)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key,)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)

    model = Decoder()
    params, state = init(model, init_key, p_sample, print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda *x: forward(model, *x)[0])
    train_metrics = ('loss',)

    train_loop(
        'decoder', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
