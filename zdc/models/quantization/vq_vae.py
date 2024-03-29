from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Patches, PatchEncoder, Reshape, TransformerBlock, VectorQuantizerProjection
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float

    @nn.compact
    def __call__(self, img, training=True):
        x = jnp.pad(img, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x = Patches(patch_size=8)(x)
        x = PatchEncoder(x.shape[1], self.hidden_dim, positional_encoding=True)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate)(x, training=training)

        x = Reshape((6, 6, self.hidden_dim))(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.embedding_dim)(x)

        return x


class Decoder(nn.Module):
    embedding_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float

    @nn.compact
    def __call__(self, z, training=True):
        x = Reshape((6 * 6, self.embedding_dim))(z)
        x = PatchEncoder(x.shape[1], self.hidden_dim, positional_encoding=True)(x)

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, 4 * self.hidden_dim, self.drop_rate)(x, training=training)

        x = Reshape((6, 6, self.hidden_dim))(x)
        x = nn.ConvTranspose(1, kernel_size=(8, 8), strides=(8, 8), padding='SAME')(x)
        x = jax.nn.relu(x + 0.5)
        x = x[:, 2:-2, 2:-2, :]

        return x


class VQVAE(nn.Module):
    num_embeddings: int = 512
    embedding_dim: int = 256
    projection_dim: int = 32
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: tuple = 6
    drop_rate: float = 0.1

    def setup(self):
        self.encoder = Encoder(self.embedding_dim, self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)
        self.decoder = Decoder(self.embedding_dim, self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)
        self.quantizer = VectorQuantizerProjection(self.num_embeddings, self.embedding_dim, self.projection_dim)

    def __call__(self, img, training=True):
        encoded = self.encoder(img, training=training)
        discrete, quantized = self.quantizer(encoded)
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

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

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
