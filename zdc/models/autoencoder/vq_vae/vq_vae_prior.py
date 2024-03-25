from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import TransformerBlock, Concatenate
from zdc.models.autoencoder.vq_vae.vq_vae import VQVAE
from zdc.models.autoencoder.vq_vae.vq_vae_cond import VQCond
from zdc.utils.data import load, batches
from zdc.utils.losses import xentropy_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule, load_model
from zdc.utils.train import train_loop


class Transformer(nn.Module):
    vocab_size: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float
    decode: bool

    @nn.compact
    def __call__(self, c, x, positions, mask, training=True):
        c = nn.Embed(self.vocab_size, self.hidden_dim)(c)
        x = nn.Embed(self.vocab_size, self.hidden_dim)(x)
        pos_emb = nn.Embed(self.seq_len, self.hidden_dim)(positions)
        x = x + pos_emb

        x = Concatenate(axis=1)(c, x)
        x = nn.LayerNorm()(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate, self.decode)(x, mask, training=training)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.vocab_size)(x)

        return x


class VQPrior(nn.Module):
    vocab_size: int = 512
    seq_len: int = 6 * 6 - 1
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1
    decode: bool = False

    @nn.compact
    def __call__(self, c, x, training=True):
        if self.decode and c is not None:
            positions = jnp.array([], dtype=int)
            x = jnp.zeros((c.shape[0], 0), dtype=int)
            mask = None
        elif self.decode and x is not None:
            is_initialized = self.has_variable('cache', 'cache_index')
            cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=int))

            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
            else:
                i = jnp.array(0, dtype=int)

            positions = i
            c = jnp.zeros((x.shape[0], 0), dtype=int)
            mask = None
        else:
            positions = jnp.arange(x.shape[1])
            mask = nn.make_causal_mask(jnp.concatenate([c, x], axis=1))

        return Transformer(
            self.vocab_size, self.seq_len, self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate, self.decode
        )(c, x, positions, mask, training=training)


def loss_fn(params, state, key, c, x, y, model):
    logits, state = forward(model, params, state, key, c, x)
    logits, y = logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
    y = jax.nn.one_hot(y, logits.shape[-1])
    loss = xentropy_loss(logits, y)
    perplexity = jnp.exp(loss)
    return loss, (state, loss, perplexity)


def eval_fn(generated, *dataset):
    _, _, y = dataset
    generated, y = generated.reshape(-1, generated.shape[-1]), y.reshape(-1)
    y = jax.nn.one_hot(y, generated.shape[-1])
    loss = xentropy_loss(generated, y)
    perplexity = jnp.exp(loss)
    return loss, perplexity


def generate_fn(params, state, cache, key, c, x, y, model):
    state['cache'] = cache
    generated = jnp.empty(y.shape + (model.vocab_size,), dtype=float)

    for i in range(c.shape[1]):
        key, subkey = jax.random.split(key)
        logits, state = forward(model, params, state, subkey, c[:, i][:, None], None, False)
        generated = generated.at[:, i].set(logits[:, 0])

    for i in range(x.shape[1]):
        key, subkey = jax.random.split(key)
        logits, state = forward(model, params, state, subkey, None, x[:, i][:, None], False)
        generated = generated.at[:, c.shape[1] + i].set(logits[:, 0])

    return generated


def tokenize_fn(key, x, batch_size, model_fn):
    tokenized = []

    for batch in batches(x, batch_size=batch_size):
        key, subkey = jax.random.split(key)
        _, discrete = jnp.where(model_fn(subkey, *batch))
        tokenized.append(discrete.reshape(batch[0].shape[0], -1))

    return jnp.concatenate(tokenized)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, r_key, p_key, train_key = jax.random.split(key, 4)

    batch_size = 256

    vq_vae, vq_vae_cond = VQVAE(), VQCond()
    vq_vae_variables = load_model('checkpoints/vq_vae/epoch_100.pkl.lz4')
    vq_vae_cond_variables = load_model('checkpoints/vq_vae_cond/epoch_100.pkl.lz4')

    vq_vae_fn = jax.jit(lambda *args: forward(vq_vae, *vq_vae_variables, *args, False)[0][2])
    vq_vae_cond_fn = jax.jit(lambda *args: forward(vq_vae_cond, *vq_vae_cond_variables, *args, False)[0][2])

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../../data', 'standard')
    r_train, r_val, r_test = jax.tree_map(lambda x: tokenize_fn(r_key, x, batch_size, vq_vae_fn), (r_train, r_val, r_test))
    c_train, c_val, c_test = jax.tree_map(lambda x: tokenize_fn(p_key, x, batch_size, vq_vae_cond_fn), (p_train, p_val, p_test))
    x_train, x_val, x_test = jax.tree_map(lambda x: x[:, :-1], (r_train, r_val, r_test))
    y_train, y_val, y_test = jax.tree_map(lambda c, x: jnp.concatenate((c, x), axis=1)[:, 1:], (c_train, c_val, c_test), (r_train, r_val, r_test))

    c_val, c_test, x_val, x_test, y_val, y_test = jax.tree_map(lambda x: x[:-(x.shape[0] % batch_size)], (c_val, c_test, x_val, x_test, y_val, y_test))

    model, model_gen = VQPrior(), VQPrior(decode=True)
    params, state = init(model, init_key, c_train[:5], x_train[:5], print_summary=True)
    cache = model_gen.init({'params': jax.random.PRNGKey(0)}, None, jnp.zeros((batch_size, y_train.shape[1]), dtype=int))['cache']

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    gen_fn = jax.jit(lambda params, state, key, *x: generate_fn(params, state, cache, key, *x, model=model_gen))
    metrics = ('loss', 'perplexity')

    train_loop(
        'vq_vae_prior', train_fn, eval_fn, gen_fn, (c_train, x_train, y_train), (c_val, x_val, y_val), (c_test, x_test, y_test),
        metrics, metrics, params, state, opt_state, train_key, epochs=40, batch_size=batch_size, n_rep=1
    )
