from functools import partial

import jax
import optax

from zdc.models.autoencoder.variational import VAE, VAEGen, loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = VAE(), VAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    opt = opt_with_cosine_schedule(partial(optax.sgd, momentum=0.9), 0.02)
    adv_opt = optax.chain(optax.contrib.normalize(), optax.sgd(0.05))
    optimizer = optax.contrib.sam(opt, adv_opt, sync_period=2, opaque_mode=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'sam_optimizer', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
