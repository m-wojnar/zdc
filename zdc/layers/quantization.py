import jax
import jax.numpy as jnp
from flax import linen as nn


class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int

    def setup(self):
        self.codebook = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self, x):
        x_flatten = x.reshape(-1, self.embedding_dim)

        distances = (
            jnp.sum(x_flatten ** 2, axis=1, keepdims=True) +
            -2 * jnp.dot(x_flatten, self.codebook.embedding.T) +
            jnp.sum(self.codebook.embedding ** 2, axis=1)
        )

        discrete = jnp.argmin(distances, axis=1)
        discrete = jax.nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.embedding)
        quantized = quantized.reshape(x.shape)

        return discrete, quantized


class VectorQuantizerProjection(nn.Module):
    num_embeddings: int
    embedding_dim: int
    projection_dim: int

    def setup(self):
        self.codebook = nn.Embed(self.num_embeddings, self.embedding_dim)
        self.projection = nn.Dense(self.projection_dim)

    def __call__(self, x):
        x_flatten = x.reshape(-1, self.embedding_dim)

        x_proj = self.projection(x_flatten)
        codebook_proj = self.projection(self.codebook.embedding)

        distances = (
            jnp.sum(x_proj ** 2, axis=1, keepdims=True) +
            -2 * jnp.dot(x_proj, codebook_proj.T) +
            jnp.sum(codebook_proj ** 2, axis=1)
        )

        discrete = jnp.argmin(distances, axis=1)
        discrete = jax.nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.embedding)
        quantized = quantized.reshape(x.shape)

        return discrete, quantized


class VectorQuantizerEMA(nn.Module):
    num_embeddings: int
    embedding_dim: int
    decay: float = 0.99
    epsilon: float = 1e-5

    def setup(self):
        self.codebook = self.variable(
            'state', 'embedding',
            jax.nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0),
            self.make_rng('zdc'), (self.num_embeddings, self.embedding_dim),
        )
        self.ema_count = self.variable(
            'stats',  'ema_count',
            jax.nn.initializers.zeros,
            self.make_rng('zdc'), self.num_embeddings
        )
        self.ema_weight = self.variable(
            'stats', 'ema_weight',
            jax.nn.initializers.variance_scaling(1 / 3, 'fan_in', 'uniform'),
            self.make_rng('zdc'), (self.num_embeddings, self.embedding_dim)
        )

    def update_embeddings(self, ema_count, ema_weight, discrete, x_flatten, batch_size):
        ema_count = ema_count * self.decay + jnp.sum(discrete, axis=0) * (1 - self.decay)
        ema_count = (ema_count + self.epsilon) / (batch_size + self.num_embeddings * self.epsilon) * batch_size
        ema_weight = ema_weight * self.decay + jnp.dot(discrete.T, x_flatten) * (1 - self.decay)
        embeddings = ema_weight / ema_count[:, None]
        return ema_count, ema_weight, embeddings

    def __call__(self, x, training=True):
        batch_size = x.shape[0]
        x_flatten = x.reshape(-1, self.embedding_dim)

        distances = (
            jnp.sum(x_flatten ** 2, axis=1, keepdims=True) +
            -2 * jnp.dot(x_flatten, self.codebook.value.T) +
            jnp.sum(self.codebook.value ** 2, axis=1)
        )

        discrete = jnp.argmin(distances, axis=1)
        discrete = jax.nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.value)
        quantized = quantized.reshape(x.shape)

        if training:
            self.ema_count.value, self.ema_weight.value, self.codebook.value = jax.lax.stop_gradient(
                self.update_embeddings(self.ema_count.value, self.ema_weight.value, discrete, x_flatten, batch_size)
            )

        return discrete, quantized
