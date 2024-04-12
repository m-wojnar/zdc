from functools import partial

import jax

from zdc.models.autoencoder.variational import Encoder, Decoder, VAE, optimizer
from zdc.utils.data import load
from zdc.utils.losses import kl_loss, mse_loss, sinkhorn_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn
from zdc.utils.wasserstein import sum_channels_parallel


def loss_fn(params, state, key, img, cond, model, sinkhorn_fn, sinkhorn_weight=0.1, kl_weight=0.7):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img, cond)
    ch_true, ch_pred = sum_channels_parallel(img), sum_channels_parallel(reconstructed)
    sinkhorn = sinkhorn_fn(ch_true, ch_pred)
    kl = kl_loss(z_mean, z_log_var)
    mse = mse_loss(img, reconstructed)
    loss = kl_weight * kl + sinkhorn_weight * sinkhorn + mse
    return loss, (state, loss, kl, sinkhorn, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VAE(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    sinkhorn_fn = sinkhorn_loss(diameter=1e-2, blur=1e-5, scaling=0.95)
    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, sinkhorn_fn=sinkhorn_fn)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss', 'kl', 'sinkhorn', 'mse')

    train_loop(
        'optimize_channels', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
