import jax
import jax.numpy as jnp
from flax import linen as nn


class VectorQuantizer(nn.Module):
    num_embeddings: int
    embedding_dim: int
    projection_dim: int = None
    normalize: bool = False

    def setup(self):
        self.codebook = nn.Embed(self.num_embeddings, self.embedding_dim)

        if self.projection_dim is not None:
            self.projection = nn.Dense(self.projection_dim, use_bias=False)

    @staticmethod
    def l2_normalize(x, axis=-1, eps=1e-12):
        return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)

    def __call__(self, x):
        x_flatten = x.reshape(-1, self.embedding_dim)
        codebook = self.codebook.embedding

        if self.projection_dim is not None:
            x_flatten = self.projection(x_flatten)
            codebook = self.projection(codebook)

        if self.normalize:
            x_flatten = self.l2_normalize(x_flatten)
            codebook = self.l2_normalize(codebook)

        distances = (
            jnp.sum(x_flatten ** 2, axis=1, keepdims=True) +
            -2 * jnp.dot(x_flatten, codebook.T) +
            jnp.sum(codebook ** 2, axis=1)
        )

        discrete = jnp.argmin(distances, axis=1)
        discrete = nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.embedding)
        quantized = quantized.reshape(x.shape[:-1] + (-1,))

        if self.normalize:
            quantized = self.l2_normalize(quantized)

        return discrete, quantized

    def quantize(self, discrete):
        return jnp.dot(discrete, self.codebook.embedding)


class VectorQuantizerEMA(VectorQuantizer):
    num_embeddings: int
    embedding_dim: int
    decay: float = 0.99
    epsilon: float = 1e-5
    projection_dim: int = None
    normalize: bool = False

    def setup(self):
        self.codebook = self.variable(
            'state', 'codebook',
            nn.initializers.variance_scaling(1.0, 'fan_in', 'normal', out_axis=0),
            self.make_rng('zdc'), (self.num_embeddings, self.embedding_dim),
        )
        self.ema_count = self.variable(
            'state',  'ema_count',
            nn.initializers.zeros,
            jax.random.PRNGKey(0), self.num_embeddings
        )
        self.ema_weight = self.variable(
            'state', 'ema_weight',
            lambda: self.codebook.value
        )

        if self.projection_dim is not None:
            self.projection = nn.Dense(self.projection_dim, use_bias=False)

    def update_embeddings(self, ema_count, ema_weight, discrete, x_flatten):
        ema_count = ema_count * self.decay + jnp.sum(discrete, axis=0) * (1 - self.decay)
        n = jnp.sum(ema_count)
        ema_count = (ema_count + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
        ema_weight = ema_weight * self.decay + jnp.dot(discrete.T, x_flatten) * (1 - self.decay)
        codebook = ema_weight / ema_count[:, None]
        return ema_count, ema_weight, codebook

    def __call__(self, x, training=True):
        x_flatten = x.reshape(-1, self.embedding_dim)
        codebook = self.codebook.value

        if self.projection_dim is not None:
            x_flatten = self.projection(x_flatten)
            codebook = self.projection(codebook)

        if self.normalize:
            x_flatten = self.l2_normalize(x_flatten)
            codebook = self.l2_normalize(codebook)

        distances = (
            jnp.sum(x_flatten ** 2, axis=1, keepdims=True) +
            -2 * jnp.dot(x_flatten, codebook.T) +
            jnp.sum(codebook ** 2, axis=1)
        )

        discrete = jnp.argmin(distances, axis=1)
        discrete = nn.one_hot(discrete, self.num_embeddings)
        quantized = jnp.dot(discrete, self.codebook.value)
        quantized = quantized.reshape(x.shape)

        if self.normalize:
            quantized = self.l2_normalize(quantized)

        if training:
            self.ema_count.value, self.ema_weight.value, self.codebook.value = jax.lax.stop_gradient(
                self.update_embeddings(self.ema_count.value, self.ema_weight.value, discrete, x.reshape(-1, self.embedding_dim))
            )

        return discrete, quantized

    def quantize(self, discrete):
        return jnp.dot(discrete, self.codebook.value)
