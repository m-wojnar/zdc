from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import UpSample, ConvBlock
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    @nn.compact
    def __call__(self, img, training=True):
        x = nn.Conv(32, kernel_size=(4, 4), strides=(2, 2))(img)
        x = nn.Conv(64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.Conv(128, kernel_size=(4, 4), strides=(2, 2))(x)
        return x


class Decoder(nn.Module):
    @nn.compact
    def __call__(self, z, training=True):
        x = UpSample()(z)
        x = ConvBlock(128, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(64, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = UpSample()(x)
        x = ConvBlock(32, kernel_size=4, use_bn=True, negative_slope=0.)(x, training=training)
        x = nn.Conv(1, kernel_size=(5, 5), padding='valid')(x)
        x = nn.relu(x)
        return x


class VQVAE(nn.Module):
    num_embeddings: int = 512
    embedding_dim: int = 128

    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.codebook = nn.Embed(num_embeddings=self.num_embeddings, features=self.embedding_dim)

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

    r_train, _, _, p_train, _, _ = load('../../../data', 'standard')
    empty_dataset = (jnp.zeros((0, 0)),)

    model = VQVAE()
    params, state = init(model, init_key, r_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, commitment_cost=0.25)))
    train_metrics = ('loss', 'mse', 'e_loss', 'q_loss', 'perplexity')

    train_loop(
        'vq_vae', train_fn, None, (r_train, p_train), empty_dataset, empty_dataset,
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
