from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.models.quantization.vq_vae import VQVAE
from zdc.models.quantization.vq_vae_cond import Encoder as CondEncoder, Decoder as CondDecoder
from zdc.models.quantization.vq_vae_prior import Transformer, eval_fn, generate_fn, loss_fn, tokenize_fn
from zdc.utils.data import load
from zdc.utils.nn import init, forward, gradient_step, load_model, opt_with_cosine_schedule
from zdc.utils.train import train_loop


optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adamw, b1=0.7, b2=0.9, eps=7e-9, weight_decay=0.03),
    peak_value=1e-4,
    pct_start=0.1,
    div_factor=20,
    final_div_factor=450,
    epochs=100,
    batch_size=256
)


class iGPT(nn.Module):
    vocab_size: int
    seq_len: int = 44 * 44 - 1
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 6
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


def dedigitize(x, bins, key):
    batch_size = x.shape[0]
    x = x.reshape(-1)
    lval, rval = bins[x - 1], bins[x]
    x = jax.random.uniform(key, x.shape, minval=lval, maxval=rval)
    x = x.reshape(batch_size, 44, 44, 1)
    return x


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key, p_key = jax.random.split(key, 3)

    batch_size = 64
    vocab_size = 256

    vq_vae_cond = VQVAE(CondEncoder, CondDecoder, embedding_dim=64, projection_dim=16)
    vq_vae_cond_variables = load_model('../quantization/checkpoints/vq_vae_cond/epoch_100.pkl.lz4')
    vq_vae_cond_fn = jax.jit(lambda *args: forward(vq_vae_cond, *vq_vae_cond_variables, *args, False)[0][2])

    r_train, r_val, r_test, p_train, p_val, p_test = load()
    r_train, r_val, r_test = jax.tree_map(lambda x: x.reshape(x.shape[0], -1), (r_train, r_val, r_test))

    vals = jnp.exp(r_train.reshape(-1)) - 1
    bins = jnp.unique(jnp.round(vals, 0))
    bins = jnp.log(bins + 1)

    c_train, c_val, c_test = jax.tree_map(lambda x: tokenize_fn(p_key, x, batch_size, vq_vae_cond_fn), (p_train, p_val, p_test))
    r_train, r_val, r_test = jax.tree_map(lambda x: jnp.digitize(x, bins), (r_train, r_val, r_test))
    x_train, x_val, x_test = jax.tree_map(lambda x: x[:, :-1], (r_train, r_val, r_test))
    y_train, y_val, y_test = jax.tree_map(lambda c, x: jnp.concatenate((c, x), axis=1)[:, 1:], (c_train, c_val, c_test), (r_train, r_val, r_test))

    c_val, c_test, x_val, x_test, y_val, y_test = jax.tree_map(lambda x: x[:-(x.shape[0] % batch_size)], (c_val, c_test, x_val, x_test, y_val, y_test))

    model, model_gen = iGPT(vocab_size), iGPT(vocab_size, decode=True)
    params, state = init(model, init_key, c_train[:5], x_train[:5], print_summary=True)
    cache = model_gen.init({'params': jax.random.PRNGKey(0)}, None, jnp.zeros((batch_size, y_train.shape[1]), dtype=int))['cache']
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    gen_fn = jax.jit(lambda params, state, key, *x: generate_fn(params, state, cache, key, *x, model=model_gen))
    metrics = ('loss', 'perplexity')

    train_loop(
        'igpt', train_fn, eval_fn, generate_fn, (c_train, x_train, y_train), (c_val, x_val, y_val), (c_test, x_test, y_test),
        metrics, metrics, params, state, opt_state, train_key, epochs=25
    )
