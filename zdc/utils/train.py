import os

import jax
import jax.numpy as jnp
from tqdm import trange

from zdc.utils.data import batches, get_samples
from zdc.utils.losses import mae_loss, mse_loss, wasserstein_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import save_model
from zdc.utils.wasserstein import sum_channels_parallel


def eval_fn(responses, generated):
    mse = mse_loss(responses, generated)
    ch_true, ch_pred = sum_channels_parallel(responses), sum_channels_parallel(generated)
    mae = mae_loss(ch_true, ch_pred) / 5
    wasserstein = wasserstein_loss(ch_true, ch_pred)
    return mse, mae, wasserstein


def train_loop(
        name, train_fn, generate_fn, train_dataset, val_dataset, test_dataset,
        train_metrics, params, state, opt_state, key, epochs, batch_size, n_rep=5
):
    metrics = Metrics(job_type='train', name=name)
    eval_metrics = ('mse', 'mae', 'wasserstein')

    os.makedirs(f'checkpoints/{name}', exist_ok=True)

    train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 5)
    r_sample, p_sample, *_ = get_samples(*train_dataset)

    for epoch in trange(epochs, desc='Epochs'):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(*train_dataset, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, (state, *losses) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add(dict(zip(train_metrics, losses)), 'train')

        metrics.log(epoch)
        responses, generated = [], []

        for batch in batches(*val_dataset, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            r_batch, p_batch, *_ = batch

            for _ in range(n_rep):
                val_key, subkey = jax.random.split(val_key)
                responses.append(r_batch)
                generated.append(generate_fn(params, state, subkey, p_batch))

        if len(responses) > 0:
            responses, generated = jnp.concatenate(responses), jnp.concatenate(generated)
            metrics.add(dict(zip(eval_metrics, eval_fn(responses, generated))), 'val')
            metrics.log(epoch)

            plot_key, subkey = jax.random.split(plot_key)
            metrics.plot_responses(r_sample, generate_fn(params, state, subkey, p_sample), epoch)

        save_model(params, state, f'checkpoints/{name}/epoch_{epoch + 1}.pkl.lz4')

    responses, generated = [], []

    for batch in batches(*test_dataset, batch_size=batch_size):
        r_batch, p_batch, *_ = batch

        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            responses.append(r_batch)
            generated.append(generate_fn(params, state, subkey, p_batch))

    if len(responses) > 0:
        responses, generated = jnp.concatenate(responses), jnp.concatenate(generated)
        metrics.add(dict(zip(eval_metrics, eval_fn(responses, generated))), 'test')
        metrics.log(epochs)
