import time

import jax

from zdc.architectures.vit import Encoder, Decoder
from zdc.models.autoencoder.variational import VAE
from zdc.utils.data import load, batches
from zdc.utils.metrics import Metrics
from zdc.utils.nn import load_model
from zdc.utils.train import default_generate_fn


if __name__ == '__main__':
    test_key = jax.random.PRNGKey(42)

    batch_size, n_rep = 256, 5
    _, _, r_test, _, _, p_test = load()
    r_test, p_test = jax.tree.map(lambda x: x[:len(x) - len(x) % batch_size], (r_test, p_test))

    model = VAE(Encoder, Decoder)
    params, state = load_model('../models/autoencoder/checkpoints/variational/epoch_100.pkl.lz4')
    generate_fn = jax.jit(default_generate_fn(model))

    batch_sample = next(batches(r_test, p_test, batch_size=batch_size))
    generate_fn(params, state, test_key, *batch_sample).block_until_ready()

    start = time.perf_counter()

    for batch in batches(r_test, p_test, batch_size=batch_size):
        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            _ = generate_fn(params, state, subkey, *batch)

    end = time.perf_counter()

    metrics = Metrics(job_type='train', name='time_variational')
    metrics.add({
        'time': end - start,
        'n_samples': n_rep * len(p_test),
        'time_per_sample': (end - start) / (n_rep * len(p_test))
    }, 'test')
    metrics.log(0)
