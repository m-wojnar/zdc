from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.layers import Concatenate, Flatten, Patches, PatchEncoder, PatchExpand, Reshape, Sampling, TransformerBlock
from zdc.models.autoencoder.variational import loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop, default_generate_fn


class Encoder(nn.Module):
    latent_dim: int
    num_heads: int
    drop_rate: float
    embedding_dim: int
    depth: int

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = Patches(patch_size=4)(img)
        x = PatchEncoder(x.shape[1], self.embedding_dim, positional_encoding=True)(x)

        c = nn.Dense(self.embedding_dim)(cond)
        c = Reshape((1, self.embedding_dim))(c)
        x = Concatenate(axis=1)(c, x)

        for _ in range(self.depth):
            x = TransformerBlock(self.num_heads, 4 * self.embedding_dim, self.drop_rate)(x, training=training)

        x = Flatten()(x)
        x = nn.Dense(2 * self.latent_dim)(x)
        x = nn.relu(x)
        z_mean = nn.Dense(self.latent_dim)(x)
        z_log_var = nn.Dense(self.latent_dim)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    num_heads: int
    drop_rate: float
    embedding_dim: int
    depth: int

    @nn.compact
    def __call__(self, z, cond, training=True):
        x = nn.Dense(11 * 11)(z)
        x = Reshape((11 * 11, 1))(x)
        x = PatchEncoder(11 * 11, self.embedding_dim, positional_encoding=True)(x)

        c = nn.Dense(self.embedding_dim)(cond)
        c = Reshape((1, self.embedding_dim))(c)
        x = Concatenate(axis=1)(c, x)

        for _ in range(self.depth):
            x = TransformerBlock(self.num_heads, 4 * self.embedding_dim, self.drop_rate)(x, training=training)

        x = x[:, 1:, :]
        x = PatchExpand(h=11, w=11)(x)
        x = PatchExpand(h=22, w=22)(x)
        x = Reshape((44, 44, self.embedding_dim // 4))(x)
        x = nn.Dense(1)(x)
        x = nn.relu(x)
        return x


class ViTVAE(nn.Module):
    latent_dim: int = 10
    num_heads: int = 4
    drop_rate: float = 0.2
    embedding_dim: int = 64
    depth: int = 4

    @nn.compact
    def __call__(self, img, cond, training=True):
        z_mean, z_log_var, z = Encoder(self.latent_dim, self.num_heads, self.drop_rate, self.embedding_dim, self.depth)(img, cond, training=training)
        reconstructed = Decoder(self.num_heads, self.drop_rate, self.embedding_dim, self.depth)(z, cond, training=training)
        return reconstructed, z_mean, z_log_var


class ViTVAEGen(ViTVAE):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.latent_dim))
        return Decoder(self.num_heads, self.drop_rate, self.embedding_dim, self.depth)(z, cond, training=False)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = ViTVAE(), ViTVAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(default_generate_fn(model_gen))
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'vit', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
