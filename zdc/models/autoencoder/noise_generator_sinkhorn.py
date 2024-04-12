from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.autoencoder.noise_generator_mse import AutoencoderNG, Encoder, Decoder, optimizer
from zdc.utils.data import load
from zdc.utils.losses import mse_loss, sinkhorn_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


def loss_fn(params, state, key, img, cond, model, sinkhorn_fn, sinkhorn_weight=20.0):
    (reconstructed, enc, le), state = forward(model, params, state, key, img, cond)
    sinkhorn = sinkhorn_fn(jnp.concatenate([enc, cond], axis=1), jnp.concatenate([le, cond], axis=1))
    mse = mse_loss(img, reconstructed)
    loss = sinkhorn_weight * sinkhorn + mse
    return loss, (state, loss, sinkhorn, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = AutoencoderNG(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    sinkhorn_fn = sinkhorn_loss(diameter=1e-2, blur=1e-5, scaling=0.95)
    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, sinkhorn_fn=sinkhorn_fn)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss', 'sinkhorn', 'mse')

    train_loop(
        'noise_encoder_sinkhorn', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
