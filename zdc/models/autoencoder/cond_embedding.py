from functools import partial

import jax
import jax.numpy as jnp
from flax import linen as nn

from zdc.architectures.vit import Encoder, Decoder, optimizer
from zdc.layers import Concatenate
from zdc.models.autoencoder.variational import VAE, loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


class ParticleEmbedding(nn.Module):
    num_embeddings: int
    embedding_dim: int

    @nn.compact
    def __call__(self, cond):
        cond, cat = cond[..., :-1], cond[..., -1].astype(int)
        emb = nn.Embed(self.num_embeddings, self.embedding_dim)(cat)
        return Concatenate()(cond, emb)


class EmbeddingVAE(nn.Module):
    encoder_type: nn.Module
    decoder_type: nn.Module
    num_embeddings: int = 21
    embedding_dim: int = 32

    def setup(self):
        self.embedding = ParticleEmbedding(self.num_embeddings, self.embedding_dim)
        self.vae = VAE(self.encoder_type, self.decoder_type)

    def __call__(self, img, cond, training=True):
        cond = self.embedding(cond)
        return self.vae(img, cond, training=training)

    def gen(self, cond):
        cond = self.embedding(cond)
        return self.vae.gen(cond)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load(load_pdgid=True)
    ps = (p_train, p_val, p_test)

    pdgids = jax.tree_map(lambda x: x[..., -1], ps)
    pdgids = jnp.concatenate(pdgids, axis=0)
    pdgids = jnp.unique(pdgids)
    pdgids_map = {pid.item(): cat for pid, cat in zip(pdgids, range(len(pdgids)))}

    categorical = jax.tree_map(lambda x: jnp.array(jax.tree_map(pdgids_map.get, x[..., -1].tolist())), ps)
    p_train, p_val, p_test = jax.tree_map(lambda x, y: jnp.concatenate([x[..., :-1], y[..., None]], axis=-1), ps, categorical)

    model = EmbeddingVAE(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'cond_embedding', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key, load_pdgid=True
    )
