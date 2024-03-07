from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.models.autoencoder.variational import Decoder as DecoderBlock
from zdc.utils.data import get_samples, load
from zdc.utils.losses import mse_loss, mae_loss, wasserstein_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop
from zdc.utils.wasserstein import sum_channels_parallel


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return DecoderBlock()(z, cond, training=training)


def loss_fn(params, state, key, img, cond, model):
    reconstructed, state = forward(model, params, state, key, cond)
    loss = mse_loss(img, reconstructed)
    return loss, (state, loss)


def eval_fn(params, state, key, img, cond, model, n_reps=5):
    def _eval_fn(subkey):
        reconstructed, _ = forward(model, params, state, subkey, cond, False)
        ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
        mse = mse_loss(img, reconstructed)
        mae = mae_loss(ch_true, ch_pred) / 5
        wasserstein = wasserstein_loss(ch_true, ch_pred)
        return mse, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


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
    eval_fn = jax.jit(partial(eval_fn, model=model))
    plot_fn = jax.jit(lambda *x: forward(model, *x)[0])

    train_metrics = ('loss',)
    eval_metrics = ('loss', 'mae', 'wasserstein')

    train_loop(
        'decoder', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
