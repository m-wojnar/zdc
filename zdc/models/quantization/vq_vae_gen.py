from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.quantization.vq_vae import VQVAE, Encoder as ImgEncoder, Decoder as ImgDecoder
from zdc.models.quantization.vq_vae_cond import Encoder as CondEncoder, Decoder as CondDecoder
from zdc.models.quantization.vq_vae_prior import VQPrior, tokenize_fn
from zdc.utils.data import load, batches, get_samples
from zdc.utils.metrics import Metrics
from zdc.utils.nn import forward, load_model
from zdc.utils.train import default_eval_fn


def select_top_k(logits, k):
    masked = jnp.argsort(logits, axis=-1)[..., :-k]
    return logits.at[jnp.arange(logits.shape[0])[:, None], masked].set(-jnp.inf)


def select_top_p(logits, p):
    sorted_logits = jnp.sort(logits, axis=-1, kind='stable')
    sorted_indices = jnp.argsort(logits, axis=-1, kind='stable')
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

    masked = jnp.where(cumulative_probs < p, sorted_logits, -jnp.inf)
    output = jnp.empty_like(masked)
    output = output.at[jnp.arange(logits.shape[0])[:, None], sorted_indices].set(masked)

    return output


def generate_prior_fn(model, params, state, cache, key, c, temperature=1.0, top_k=None, top_p=None):
    state['cache'] = cache

    for i in range(c.shape[1]):
        key, subkey = jax.random.split(key)
        logits, state = forward(model, params, state, subkey, c[:, i][:, None], None, False)

        if top_k is not None:
            logits = select_top_k(logits, top_k)
        elif top_p is not None:
            logits = select_top_p(logits, top_p)

    key, subkey = jax.random.split(key)
    next_token = jax.random.categorical(subkey, logits / temperature, axis=-1)

    generated = jnp.empty((c.shape[0], 6 * 6), dtype=float)
    generated = generated.at[:, 0].set(next_token[:, 0])

    for i in range(6 * 6 - 1):
        key, subkey_forward, subkey_categorical = jax.random.split(key, 3)
        logits, state = forward(model, params, state, subkey_forward, None, next_token, False)

        if top_k is not None:
            logits = select_top_k(logits, top_k)
        elif top_p is not None:
            logits = select_top_p(logits, top_p)

        next_token = jax.random.categorical(subkey_categorical, logits / temperature, axis=-1)
        generated = generated.at[:, i + 1].set(next_token[:, 0])

    return generated


def generate_fn(key, x, c, vq_vae, vq_vae_prior):
    prior_key, decoder_key = jax.random.split(key)
    vq_vae, vq_vae_variables = vq_vae
    vq_vae_prior, vq_vae_prior_variables = vq_vae_prior

    cache = vq_vae_prior.init({'params': jax.random.PRNGKey(0)}, None, jnp.zeros((c.shape[0], 6 * 6 + c.shape[1] - 1), dtype=int))['cache']
    generated = generate_prior_fn(vq_vae_prior, *vq_vae_prior_variables, cache, prior_key, c)
    generated, _ = forward(vq_vae, *vq_vae_variables, decoder_key, generated, method='gen')

    return generated


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, p_key, test_key, plot_key = jax.random.split(key, 4)

    batch_size = 256
    n_rep = 5

    vq_vae = VQVAE(ImgEncoder, ImgDecoder)
    vq_vae_cond = VQVAE(CondEncoder, CondDecoder, embedding_dim=64, projection_dim=16)
    vq_vae_prior = VQPrior(decode=True)
    vq_vae_variables = load_model('checkpoints/vq_vae/epoch_100.pkl.lz4')
    vq_vae_cond_variables = load_model('checkpoints/vq_vae_cond/epoch_100.pkl.lz4')
    vq_vae_prior_variables = load_model('checkpoints/vq_vae_prior/epoch_100.pkl.lz4')

    vq_vae_cond_fn = jax.jit(lambda *args: forward(vq_vae_cond, *vq_vae_cond_variables, *args, False)[0][2])

    r_train, _, r_test, p_train, _, p_test = load()
    r_sample, p_sample = get_samples()
    c_test, c_sample = jax.tree_map(lambda x, k: tokenize_fn(k, x, batch_size, vq_vae_cond_fn), (p_test, p_sample), tuple(jax.random.split(p_key)))

    gen_fn = jax.jit(partial(generate_fn, vq_vae=(vq_vae, vq_vae_variables), vq_vae_prior=(vq_vae_prior, vq_vae_prior_variables)))

    metrics = Metrics(job_type='train', name='vq_vae_gen')
    eval_metrics = ('mse', 'mae', 'wasserstein')
    generated, original = [], []

    for batch in batches(r_test, c_test, batch_size=batch_size):
        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            generated.append(gen_fn(subkey, *batch))
            original.append(batch)

    generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
    metrics.add(dict(zip(eval_metrics, default_eval_fn(generated, *original))), 'test')
    metrics.plot_responses(r_sample, gen_fn(plot_key, r_sample, c_sample), 0)
    metrics.log(0)
