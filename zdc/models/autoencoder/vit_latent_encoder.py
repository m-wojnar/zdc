from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.layers import Flatten, Patches, PatchEncoder, PatchExpand, Reshape, TransformerEncoderBlock
from zdc.models.autoencoder.latent_encoder import loss_fn, LatentEncoder
from zdc.utils.data import get_samples, load
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    latent_dim: int
    num_heads: int
    drop_rate: float
    embedding_dim: int
    depth: int

    @nn.compact
    def __call__(self, img, training=True):
        x = Patches(patch_size=4)(img)
        x = PatchEncoder(x.shape[1], self.embedding_dim, positional_encoding=True)(x)

        for _ in range(self.depth):
            x = TransformerEncoderBlock(self.num_heads, 4 * self.embedding_dim, self.drop_rate)(x, training=training)

        x = Flatten()(x)
        x = nn.Dense(self.latent_dim)(x)
        x = nn.relu(x)
        return x


class Decoder(nn.Module):
    num_heads: int
    drop_rate: float
    embedding_dim: int
    depth: int

    @nn.compact
    def __call__(self, cond, training=True):
        x = nn.Dense(11 * 11)(cond)
        x = Reshape((11 * 11, 1))(x)
        x = PatchEncoder(11 * 11, self.embedding_dim, positional_encoding=True)(x)

        for _ in range(self.depth):
            x = TransformerEncoderBlock(self.num_heads, 4 * self.embedding_dim, self.drop_rate)(x, training=training)

        x = PatchExpand(h=11, w=11)(x)
        x = PatchExpand(h=22, w=22)(x)
        x = Reshape((44, 44, self.embedding_dim // 4))(x)
        x = nn.Dense(1)(x)
        x = nn.relu(x)
        return x


class LEViT(nn.Module):
    latent_dim: int = 20
    noise_dim: int = 10
    num_heads: int = 4
    drop_rate: float = 0.2
    embedding_dim: int = 64
    depth: int = 4

    @nn.compact
    def __call__(self, img, cond, training=True):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.latent_dim))
        enc = Encoder(self.latent_dim, self.num_heads, self.drop_rate, self.embedding_dim, self.depth)(img, training=training)
        le = LatentEncoder()(z, cond, training=training)
        reconstructed = Decoder(self.num_heads, self.drop_rate, self.embedding_dim, self.depth)(le, training=training)
        return reconstructed, enc, le


class LEViTGen(LEViT):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.latent_dim))
        le = LatentEncoder()(z, cond, training=False)
        return Decoder(self.num_heads, self.drop_rate, self.embedding_dim, self.depth)(le, training=False)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)

    model, model_gen = LEViT(), LEViTGen()
    params, state = init(model, init_key, r_sample, p_sample, print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, enc_weight=10.)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'mse_enc', 'mse_rec')

    train_loop(
        'vit_latent_encoder', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
