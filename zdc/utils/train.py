import os

import jax
import jax.numpy as jnp
from tqdm import trange

from zdc.models import RESPONSE_SHAPE
from zdc.utils.data import batches, get_samples
from zdc.utils.losses import mae_loss, mse_loss, wasserstein_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import save_model, forward
from zdc.utils.wasserstein import sum_channels_parallel


def default_eval_fn(generated, *dataset):
    img, *_ = dataset
    img, generated = jnp.exp(img) - 1, jnp.exp(generated) - 1

    rmse = jnp.sqrt(mse_loss(img, generated))
    ch_true, ch_pred = sum_channels_parallel(img, apply_exp=False), sum_channels_parallel(generated, apply_exp=False)
    mae = mae_loss(ch_true, ch_pred) / 5
    wasserstein = wasserstein_loss(ch_true, ch_pred)

    return rmse, mae, wasserstein


def default_generate_fn(model):
    def generate_fn(params, state, key, *x):
        return forward(model, params, state, key, x[1], method='gen')[0]

    return generate_fn


def train_loop(
        name, train_fn, eval_fn, generate_fn, train_dataset, val_dataset, test_dataset,
        train_metrics, eval_metrics, params, state, opt_state, key, epochs=100, batch_size=256, n_rep=5, load_pdgid=False
):
    if eval_fn is None:
        eval_fn = default_eval_fn
        eval_metrics = ('rmse', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name=name)
    os.makedirs(f'checkpoints/{name}', exist_ok=True)

    train_key, val_key, test_key, shuffle_key, plot_key = jax.random.split(key, 5)
    samples = get_samples(load_pdgid=load_pdgid)

    eval_fn = jax.jit(eval_fn)

    for epoch in trange(epochs, desc='Epochs'):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(*train_dataset, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, (state, *losses) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add(dict(zip(train_metrics, losses)), 'train')

        metrics.log(epoch)
        generated, original = [], []

        for batch in batches(*val_dataset, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            for _ in range(n_rep):
                val_key, subkey = jax.random.split(val_key)
                generated.append(generate_fn(params, state, subkey, *batch))
                original.append(batch)

        generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
        metrics.add(dict(zip(eval_metrics, eval_fn(generated, *original))), 'val')
        metrics.log(epoch)

        if generated.shape[1:] == RESPONSE_SHAPE:
            plot_key, subkey = jax.random.split(plot_key)
            metrics.plot_responses(samples[0], generate_fn(params, state, subkey, *samples), epoch)

        save_model(params, state, f'checkpoints/{name}/epoch_{epoch + 1}.pkl.lz4')

    generated, original = [], []

    for batch in batches(*test_dataset, batch_size=batch_size):
        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            generated.append(generate_fn(params, state, subkey, *batch))
            original.append(batch)

    generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
    metrics.add(dict(zip(eval_metrics, eval_fn(generated, *original))), 'test')
    metrics.log(epochs)
