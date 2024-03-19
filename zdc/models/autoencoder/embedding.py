from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate
from zdc.models.autoencoder.variational import Encoder, Decoder, loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class EmbeddingVAE(nn.Module):
    num_embeddings: int
    embedding_dim: int = 16

    @nn.compact
    def __call__(self, img, cond, training=True):
        cond, cat = cond[..., :-1], cond[..., -1].astype(int)
        emb = nn.Embed(self.num_embeddings, self.embedding_dim)(cat)
        cond = Concatenate()(cond, emb)
        z_mean, z_log_var, z = Encoder()(img, cond, training=training)
        reconstructed = Decoder()(z, cond, training=training)
        return reconstructed, z_mean, z_log_var


class EmbeddingVAEGen(nn.Module):
    num_embeddings: int
    embedding_dim: int = 16

    @nn.compact
    def __call__(self, cond):
        cond, cat = cond[..., :-1], cond[..., -1].astype(int)
        emb = nn.Embed(self.num_embeddings, self.embedding_dim)(cat)
        cond = Concatenate()(cond, emb)
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 10))
        return Decoder()(z, cond, training=False)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard', load_pdgid=True)
    ps = (p_train, p_val, p_test)

    pdgids = jax.tree_map(lambda x: x[..., -1], ps)
    pdgids = jnp.concatenate(pdgids, axis=0)
    pdgids = jnp.unique(pdgids)
    pdgids_map = {pid.item(): cat for pid, cat in zip(pdgids, range(len(pdgids)))}

    categorical = jax.tree_map(lambda x: jnp.array(jax.tree_map(pdgids_map.get, x[..., -1].tolist())), ps)
    p_train, p_val, p_test = jax.tree_map(lambda x, y: jnp.concatenate([x[..., :-1], y[..., None]], axis=-1), ps, categorical)

    model, model_gen = EmbeddingVAE(len(pdgids)), EmbeddingVAEGen(len(pdgids))
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'embedding', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
