from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.layers import DenseBlock, Reshape
from zdc.models import PARTICLE_SHAPE
from zdc.models.quantization.vq_vae import loss_fn, VQVAE
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    embedding_dim: int = 64
    latent_dim: int = 2

    @nn.compact
    def __call__(self, cond, training=True):
        x = DenseBlock(self.embedding_dim, negative_slope=0.2)(cond)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = Reshape((self.latent_dim, self.embedding_dim))(x)
        return x


class Decoder(nn.Module):
    embedding_dim: int = 64
    latent_dim: int = 2

    @nn.compact
    def __call__(self, z, training=True):
        x = Reshape((self.latent_dim * self.embedding_dim,))(z)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.latent_dim * self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(self.embedding_dim, negative_slope=0.2)(x)
        x = DenseBlock(*PARTICLE_SHAPE, negative_slope=0.2)(x)
        return x


def eval_fn(generated, *dataset):
    cond, _ = dataset
    return (mse_loss(cond, generated),)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VQVAE(Encoder, Decoder, embedding_dim=64, projection_dim=16)
    params, state = init(model, init_key, p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0], False)[0][0])
    train_metrics = ('loss', 'mse', 'e_loss', 'q_loss', 'perplexity')
    eval_metrics = ('mse',)

    train_loop(
        'vq_vae_cond', train_fn, eval_fn, generate_fn, (p_train, r_train), (p_val, r_val), (p_test, r_test),
        train_metrics, eval_metrics, params, state, opt_state, train_key, n_rep=1
    )
