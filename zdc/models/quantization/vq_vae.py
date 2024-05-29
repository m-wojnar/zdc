from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.architectures.vit import Encoder, Decoder
from zdc.layers import VectorQuantizer
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adamw, b1=0.71, b2=0.88, eps=6.7e-9, weight_decay=0.031),
    peak_value=1.7e-3,
    pct_start=0.1,
    div_factor=22,
    final_div_factor=44,
    epochs=100,
    batch_size=256
)


class VQVAE(nn.Module):
    encoder_type: nn.Module
    decoder_type: nn.Module
    quantizer_type: nn.Module = VectorQuantizer
    num_embeddings: int = 256
    embedding_dim: int = 256
    projection_dim: int = 32
    normalize: bool = True

    def setup(self):
        self.encoder = self.encoder_type()
        self.pre_embedding = nn.Dense(self.embedding_dim)
        self.quantizer = self.quantizer_type(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_dim, projection_dim=self.projection_dim, normalize=self.normalize)
        self.post_embedding = nn.Dense(self.embedding_dim)
        self.decoder = self.decoder_type()

    def __call__(self, img, training=True):
        encoded = self.encoder(img, training=training)
        encoded = self.pre_embedding(encoded)

        discrete, quantized = self.quantizer(encoded)
        quantized = self.post_embedding(quantized)

        encoded = VectorQuantizer.l2_normalize(encoded) if self.normalize else encoded
        quantized_sg = encoded + jax.lax.stop_gradient(quantized - encoded)

        reconstructed = self.decoder(quantized_sg, training=training)
        return reconstructed, encoded, discrete, quantized

    def gen(self, discrete):
        discrete = nn.one_hot(discrete, self.num_embeddings)
        quantized = self.quantizer.quantize(discrete)
        quantized = quantized.reshape(-1, 6 * 6, self.embedding_dim)
        quantized = VectorQuantizer.l2_normalize(quantized) if self.normalize else quantized
        quantized = self.post_embedding(quantized)
        reconstructed = self.decoder(quantized, training=False)
        return reconstructed


def loss_fn(params, state, key, img, cond, model, commitment_cost=0.25):
    (reconstructed, encoded, discrete, quantized), state = forward(model, params, state, key, img)

    e_loss = mse_loss(jax.lax.stop_gradient(quantized), encoded)
    q_loss = mse_loss(quantized, jax.lax.stop_gradient(encoded))
    mse = mse_loss(img, reconstructed)
    loss = mse + commitment_cost * e_loss + q_loss

    avg_prob = jnp.mean(discrete, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_prob * jnp.log(avg_prob + 1e-10)))

    return loss, (state, loss, mse, e_loss, q_loss, perplexity)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VQVAE(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0], False)[0][0])
    train_metrics = ('loss', 'mse', 'e_loss', 'q_loss', 'perplexity')

    train_loop(
        'vq_vae', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
