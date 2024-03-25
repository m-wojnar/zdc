from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Patches, PatchEncoder, PatchMerge, PatchExpand, Reshape, TransformerBlock
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: tuple
    drop_rate: float

    @nn.compact
    def __call__(self, img, training=True):
        x = Patches(patch_size=4)(img)
        x = PatchEncoder(x.shape[1], self.hidden_dim, positional_encoding=True)(x)
        x = nn.LayerNorm()(x)

        for _ in range(self.num_layers[0]):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate)(x, training=training)

        x = PatchMerge(h=11, w=11)(x)

        for _ in range(self.num_layers[1]):
            x = TransformerBlock(self.num_heads, 8 * self.hidden_dim, self.drop_rate)(x, training=training)

        x = Reshape((6, 6, 2 * self.hidden_dim))(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embedding_dim)(x)

        return x


class Decoder(nn.Module):
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: tuple
    drop_rate: float

    @nn.compact
    def __call__(self, z, training=True):
        x = Reshape((6 * 6, self.embedding_dim))(z)
        x = PatchExpand(h=6, w=6)(x)
        x = Reshape((12, 12, self.embedding_dim // 2))(x)
        x = x[:, :-1, :-1, :]
        x = Reshape((11 * 11, self.embedding_dim // 2))(x)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim)(x)

        for _ in range(self.num_layers[0]):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate)(x, training=training)

        x = PatchExpand(h=11, w=11)(x)
        x = PatchExpand(h=22, w=22)(x)

        for _ in range(self.num_layers[1]):
            x = TransformerBlock(self.num_heads, self.hidden_dim, self.drop_rate)(x, training=training)

        x = Reshape((44, 44, -1))(x)
        x = nn.Dense(1)(x)

        return x


class VQVAE(nn.Module):
    num_embeddings: int = 256
    embedding_dim: int = 128
    encoder_hidden_dim: int = 128
    decoder_hidden_dim: int = 256
    num_heads: int = 4
    num_layers: tuple = (2, 2)
    drop_rate: float = 0.1

    def setup(self):
        self.encoder = Encoder(self.embedding_dim, self.encoder_hidden_dim, self.num_heads, self.num_layers, self.drop_rate)
        self.decoder = Decoder(self.embedding_dim, self.decoder_hidden_dim, self.num_heads, self.num_layers, self.drop_rate)
        self.codebook = nn.Embed(self.num_embeddings, self.embedding_dim)

    def __call__(self, img, training=True):
        encoded = self.encoder(img, training=training)
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


def loss_fn(params, state, key, img, cond, model, commitment_cost):
    (reconstructed, encoded, discrete, quantized), state = forward(model, params, state, key, img)
    e_loss = mse_loss(jax.lax.stop_gradient(quantized), encoded)
    q_loss = mse_loss(quantized, jax.lax.stop_gradient(encoded))
    mse = mse_loss(img, reconstructed)
    avg_prob = jnp.mean(discrete, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_prob * jnp.log(avg_prob + 1e-10)))
    loss = mse + commitment_cost * e_loss + q_loss
    return loss, (state, loss, mse, e_loss, q_loss, perplexity)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../../data', 'standard')

    model = VQVAE()
    params, state = init(model, init_key, r_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, commitment_cost=0.25)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0], False)[0][0])
    train_metrics = ('loss', 'mse', 'e_loss', 'q_loss', 'perplexity')

    train_loop(
        'vq_vae', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
