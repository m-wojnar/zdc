from functools import partial

import jax
import jax.numpy as jnp
import optax

from zdc.models.autoencoder.supervised import SupervisedAE, SupervisedAEGen
from zdc.utils.data import get_samples, load
from zdc.utils.losses import mse_loss, mae_loss, wasserstein_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop
from zdc.utils.wasserstein import sum_channels_parallel


def loss_fn(params, state, key, img, cond, model, cond_weight):
    (reconstructed, encoder_cond), state = forward(model, params, state, key, img)
    mse_cond = mse_loss(cond, encoder_cond)
    mse_rec = mse_loss(img, reconstructed)
    ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
    mse_ch = mse_loss(ch_true, ch_pred)
    loss = cond_weight * mse_cond + mse_rec + mse_ch
    return loss, (state, loss, mse_cond, mse_rec, mse_ch)


def eval_fn(params, state, key, img, cond, model, cond_weight, n_reps=5):
    def _eval_fn(subkey):
        (reconstructed, encoder_cond), _ = forward(model, params, state, subkey, img, False)
        ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
        mse_cond = mse_loss(cond, encoder_cond)
        mse_rec = mse_loss(img, reconstructed)
        mse_ch = mse_loss(ch_true, ch_pred)
        mae = mae_loss(ch_true, ch_pred) / 5
        wasserstein = wasserstein_loss(ch_true, ch_pred)
        return cond_weight * mse_cond + mse_rec + mse_ch, mse_cond, mse_rec, mse_ch, mae, wasserstein

    results = jax.vmap(_eval_fn)(jax.random.split(key, n_reps))
    return jnp.array(results).mean(axis=1)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)

    model, model_gen = SupervisedAE(), SupervisedAEGen()
    params, state = init(model, init_key, r_sample, print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=1.)))
    eval_fn = jax.jit(partial(eval_fn, model=model, cond_weight=1.))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('loss', 'mse_cond', 'mse_rec', 'mse_ch')
    eval_metrics = ('loss', 'mse_cond', 'mse_rec', 'mse_ch', 'mae', 'wasserstein')

    train_loop(
        'supervised_channels', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
