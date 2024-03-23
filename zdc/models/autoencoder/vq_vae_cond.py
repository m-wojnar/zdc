from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import DenseBlock, Reshape
from zdc.models import PARTICLE_SHAPE
from zdc.models.autoencoder.vq_vae import loss_fn
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    embedding_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, cond, training=True):
        x = DenseBlock(self.embedding_dim, negative_slope=0.2)(cond)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = Reshape((self.latent_dim, self.embedding_dim))(x)
        return x


class Decoder(nn.Module):
    embedding_dim: int
    latent_dim: int
    particle_dim: int
    
    @nn.compact
    def __call__(self, z, training=True):
        x = Reshape((self.latent_dim * self.embedding_dim,))(z)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.particle_dim, negative_slope=0.2)(x)
        return x


class VQVAE(nn.Module):
    num_embeddings: int = 512
    embedding_dim: int = 32
    latent_dim: int = 2

    def setup(self):
        self.encoder = Encoder(self.embedding_dim, self.latent_dim)
        self.decoder = Decoder(self.embedding_dim, self.latent_dim, *PARTICLE_SHAPE)
        self.codebook = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self, cond, training=True):
        encoded = self.encoder(cond, training=training)
        encoded_flatten = encoded.reshape((-1, self.embedding_dim))
        distances = (
            jnp.sum(encoded_flatten ** 2, axis=1, keepdims=True) +
            -2 * jnp.dot(encoded_flatten, self.codebook.embedding.T) +
            jnp.sum(self.codebook.embedding ** 2, axis=1)
        )
        discrete = jnp.argmin(distances, axis=1)
        discrete = jax.nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.embedding)
        quantized = quantized.reshape(encoded.shape)
        quantized_sg = encoded + jax.lax.stop_gradient(quantized - encoded)
        reconstructed = self.decoder(quantized_sg, training=training)
        return reconstructed, encoded, discrete, quantized


def eval_fn(generated, *dataset):
    cond, _ = dataset
    return (mse_loss(cond, generated),)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model = VQVAE()
    params, state = init(model, init_key, p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, commitment_cost=0.25)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0])[0][0])
    train_metrics = ('loss', 'mse', 'e_loss', 'q_loss', 'perplexity')

    train_loop(
        'vq_vae_cond', train_fn, eval_fn, generate_fn, (p_train, r_train), (p_val, r_val), (p_test, r_test),
        train_metrics, ('mse',), params, state, opt_state, train_key, epochs=100, batch_size=128
    )
