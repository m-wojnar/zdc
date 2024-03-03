import os
from functools import partial

import jax
import optax
from tqdm import trange

from zdc.models.autoencoder.supervised import SupervisedAE, SupervisedAEGen, loss_fn, eval_fn
from zdc.utils.data import load, batches
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model, print_model


if __name__ == '__main__':
    batch_size = 128
    cond_weight = 1.0
    n_reps = 5
    lr = 1e-4
    epochs = 100
    seed = 42

    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 6)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model, model_gen = SupervisedAE(), SupervisedAEGen()
    params, state = init(model, init_key, r_sample)
    print_model(params)

    opt = optax.adam(lr)
    adv_opt = optax.chain(optax.contrib.normalize(), optax.adam(0.02))
    optimizer = optax.contrib.sam(opt, adv_opt, sync_period=5, opaque_mode=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, cond_weight=cond_weight)))
    eval_fn = jax.jit(partial(eval_fn, model=model, cond_weight=cond_weight, n_reps=n_reps))
    eval_metrics = ('loss', 'mse_cond', 'mse_rec', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='supervised_sam')
    os.makedirs('checkpoints/supervised_sam', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(r_train, p_train, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, loss, (state, mse_cond, mse_rec) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add({'loss': loss, 'mse_cond': mse_cond, 'mse_rec': mse_rec}, 'train')

        metrics.log(epoch)

        for batch in batches(r_val, p_val, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, forward(model_gen, params, state, subkey, p_sample)[0], epoch)

        save_model(params, state, f'checkpoints/supervised_sam/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
