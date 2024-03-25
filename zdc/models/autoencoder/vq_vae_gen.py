from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from zdc.models.autoencoder.vq_vae import Decoder
from zdc.models.autoencoder.vq_vae_cond import VQCond
from zdc.models.autoencoder.vq_vae_prior import VQPrior, tokenize_fn
from zdc.utils.data import load, batches, get_samples
from zdc.utils.metrics import Metrics
from zdc.utils.nn import forward, load_model
from zdc.utils.train import default_eval_fn


class VQVAEGen(nn.Module):
    num_embeddings: int = 512
    embedding_dim: int = 128

    def setup(self):
        self.decoder = Decoder()
        self.codebook = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self, discrete):
        discrete = jax.nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.embedding)
        quantized = quantized.reshape(-1, 6, 6, self.embedding_dim)
        reconstructed = self.decoder(quantized, training=False)
        return reconstructed


def generate_prior_fn(params, state, cache, key, c, model):
    state['cache'] = cache

    for i in range(c.shape[1]):
        key, subkey = jax.random.split(key)
        logits, state = forward(model, params, state, subkey, c[:, i][:, None], None, False)

    key, subkey = jax.random.split(key)
    next_token = jax.random.categorical(subkey, logits, axis=-1)

    generated = jnp.empty((c.shape[0], 6 * 6), dtype=float)
    generated = generated.at[:, 0].set(next_token[:, 0])

    for i in range(6 * 6 - 1):
        key, subkey_forward, subkey_categorical = jax.random.split(key, 3)
        logits, state = forward(model, params, state, subkey_forward, None, next_token, False)
        next_token = jax.random.categorical(subkey_categorical, logits, axis=-1)
        generated = generated.at[:, i + 1].set(next_token[:, 0])

    return generated


def generate_fn(key, x, c, vq_vae, vq_vae_prior):
    prior_key, decoder_key = jax.random.split(key)
    vq_vae, vq_vae_params, vq_vae_state = vq_vae
    vq_vae_prior, vq_vae_prior_params, vq_vae_prior_state = vq_vae_prior

    cache = vq_vqe_prior.init({'params': jax.random.PRNGKey(0)}, None, jnp.zeros((c.shape[0], 6 * 6 + c.shape[1] - 1), dtype=int))['cache']
    generated = generate_prior_fn(vq_vae_prior_params, vq_vae_prior_state, cache, prior_key, c, vq_vae_prior)
    generated, _ = forward(vq_vae, vq_vae_params, vq_vae_state, decoder_key, generated)

    return generated


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, p_key, test_key, plot_key = jax.random.split(key, 4)

    batch_size = 256
    n_rep = 5

    vq_vqe, vq_vqe_cond, vq_vqe_prior = VQVAEGen(), VQCond(), VQPrior(decode=True)
    vq_vae_params, vq_vae_state = load_model('checkpoints/vq_vae/epoch_100.pkl.lz4')
    vq_vae_cond_params, vq_vae_cond_state = load_model('checkpoints/vq_vae_cond/epoch_100.pkl.lz4')
    vq_vae_prior_params, vq_vae_prior_state = load_model('checkpoints/vq_vae_prior/epoch_25.pkl.lz4')

    r_train, _, r_test, p_train, _, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)
    c_test, c_sample = jax.tree_map(lambda x, k: tokenize_fn(vq_vae_cond_params, vq_vae_cond_state, k, x, batch_size, vq_vqe_cond), (p_test, p_sample), tuple(jax.random.split(p_key)))

    gen_fn = jax.jit(partial(generate_fn, vq_vae=(vq_vqe, vq_vae_params, vq_vae_state), vq_vae_prior=(vq_vqe_prior, vq_vae_prior_params, vq_vae_prior_state)))

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
