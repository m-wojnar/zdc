from functools import partial

import jax
import optax

from zdc.models.autoencoder.variational import VAE, VAEGen
from zdc.utils.data import load
from zdc.utils.losses import kl_loss, mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop
from zdc.utils.wasserstein import sum_channels_parallel


def loss_fn(params, state, key, img, cond, model, cond_weight, kl_weight):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img, cond)
    ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
    mse_ch = mse_loss(ch_true, ch_pred)
    kl = kl_loss(z_mean, z_log_var)
    mse_rec = mse_loss(img, reconstructed)
    loss = kl_weight * kl + cond_weight * mse_ch + mse_rec
    return loss, (state, loss, kl, mse_ch, mse_rec)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = VAE(), VAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=0.05, kl_weight=0.7)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'kl', 'mse_ch', 'mse_rec')

    train_loop(
        'channels', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
