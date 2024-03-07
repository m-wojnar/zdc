from functools import partial

import jax
import optax

from zdc.models.autoencoder.supervised import SupervisedAE, SupervisedAEGen, loss_fn, eval_fn
from zdc.utils.data import get_samples, load
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)

    model, model_gen = SupervisedAE(), SupervisedAEGen()
    params, state = init(model, init_key, r_sample, print_summary=True)

    opt = optax.adam(1e-4)
    adv_opt = optax.chain(optax.contrib.normalize(), optax.adam(0.02))
    optimizer = optax.contrib.sam(opt, adv_opt, sync_period=5, opaque_mode=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=1.)))
    eval_fn = jax.jit(partial(eval_fn, model=model, cond_weight=1.))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('loss', 'mse_cond', 'mse_rec')
    eval_metrics = ('loss', 'mse_cond', 'mse_rec', 'mae', 'wasserstein')

    train_loop(
        'supervised_sam', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
