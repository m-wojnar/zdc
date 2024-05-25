import jax
import jax.numpy as jnp
from scipy.optimize import minimize_scalar

from zdc.models.gan.gan import GAN
from zdc.utils.data import load, batches, get_samples
from zdc.utils.losses import wasserstein_loss
from zdc.utils.metrics import Metrics
from zdc.utils.nn import load_model
from zdc.utils.train import default_eval_fn, default_generate_fn
from zdc.utils.wasserstein import sum_channels_parallel


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    train_key, test_key, plot_key = jax.random.split(key, 3)

    batch_size, n_rep = 256, 5
    r_train, _, r_test, p_train, _, p_test = load()
    r_sample, p_sample = get_samples()

    model = GAN()
    params, state = load_model('checkpoints/gan/epoch_100.pkl.lz4')
    generate_fn = jax.jit(default_generate_fn(model))
    wasserstein_fn = jax.jit(wasserstein_loss)

    generated, original = [], []

    for batch in batches(r_train, p_train, batch_size=batch_size):
        for _ in range(n_rep):
            train_key, subkey = jax.random.split(train_key)
            generated.append(generate_fn(params, state, subkey, *batch))
            original.append(batch)

    generated, original = jnp.concatenate(generated), tuple(jnp.concatenate(xs) for xs in zip(*original))
    ch_true, ch_pred = sum_channels_parallel(original[0]), sum_channels_parallel(generated)

    def objective_fn(c):
        return wasserstein_fn(ch_true, c * ch_pred)

    c_optim = minimize_scalar(objective_fn, bounds=(0.9, 1.1), method='bounded').x
    print(f'Optimal scaling factor: {c_optim:.3f}')

    metrics = Metrics(job_type='train', name='gan_postprocessing')
    eval_metrics = ('mse', 'mae', 'wasserstein')
    generated, original = [], []

    for batch in batches(r_test, p_test, batch_size=batch_size):
        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            generated.append(c_optim * generate_fn(params, state, subkey, *batch))
            original.append(batch)

    generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
    metrics.add(dict(zip(eval_metrics, default_eval_fn(generated, *original))), 'test')
    metrics.plot_responses(r_sample, generate_fn(params, state, plot_key, r_sample, p_sample), 0)
    metrics.log(0)
