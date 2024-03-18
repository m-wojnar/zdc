from functools import partial

import jax
import optax

from zdc.models.autoencoder.supervised import SupervisedAE, SupervisedAEGen, loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = SupervisedAE(), SupervisedAEGen()
    params, state = init(model, init_key, r_train[:5], print_summary=True)

    opt = optax.adam(1e-4)
    adv_opt = optax.chain(optax.contrib.normalize(), optax.adam(0.02))
    optimizer = optax.contrib.sam(opt, adv_opt, sync_period=5, opaque_mode=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=1.)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'mse_cond', 'mse_rec')

    train_loop(
        'supervised_sam', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
