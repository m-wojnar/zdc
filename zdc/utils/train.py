import os

import jax
from tqdm import trange

from zdc.utils.data import batches
from zdc.utils.metrics import Metrics
from zdc.utils.nn import save_model


def train_loop(
        name, train_fn, eval_fn, plot_fn, train_dataset, val_dataset, test_dataset, r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, key, epochs, batch_size
):
    metrics = Metrics(job_type='train', name=name)
    os.makedirs(f'checkpoints/{name}', exist_ok=True)

    train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 5)

    for epoch in trange(epochs, desc='Epochs'):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(*train_dataset, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, (state, *losses) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add(dict(zip(train_metrics, losses)), 'train')

        metrics.log(epoch)

        for batch in batches(*val_dataset, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, plot_fn(params, state, subkey, p_sample), epoch)

        save_model(params, state, f'checkpoints/{name}/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(*test_dataset, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
