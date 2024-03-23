from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Concatenate, ConvNeXtV2Embedding, ConvNeXtV2Block, ConvNeXtV2Stage, GlobalAveragePooling, Sampling, UpSample, Reshape
from zdc.models.autoencoder.variational import loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop, default_generate_fn


class Encoder(nn.Module):
    latent_dim: int = 32
    kernel_size: int = 3
    max_drop_rate: float = 0.33
    depths: tuple = (1, 1, 3, 1)
    projection_dims: tuple = (24, 48, 96, 192)
    drop_rates = [r.tolist() for r in jnp.split(jnp.linspace(0., max_drop_rate, sum(depths)), jnp.cumsum(jnp.array(depths))[:-1])]

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = ConvNeXtV2Embedding(patch_size=2, projection_dim=self.projection_dims[0])(img)

        for i, (projection_dim, drop_rates) in enumerate(zip(self.projection_dims, self.drop_rates)):
            patch_size = 2 if i > 0 else 1
            x = ConvNeXtV2Stage(patch_size, projection_dim, self.kernel_size, drop_rates)(x, training=training)

        x = GlobalAveragePooling()(x)
        x = nn.LayerNorm()(x)
        x = Concatenate()(x, cond)
        x = nn.Dense(2 * self.latent_dim)(x)
        x = nn.gelu(x)

        z_mean = nn.Dense(self.latent_dim)(x)
        z_log_var = nn.Dense(self.latent_dim)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    kernel_size: int = 3
    decoder_dim: int = 128

    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate()(z, cond)
        x = nn.Dense(6 * 6 * self.decoder_dim)(x)
        x = Reshape((6, 6, self.decoder_dim))(x)

        for _ in range(3):
            x = UpSample()(x)
            x = ConvNeXtV2Block(self.decoder_dim, self.kernel_size)(x, training=training)

        x = nn.Conv(1, kernel_size=(5, 5), padding='valid')(x)
        x = nn.relu(x)
        return x


class ConvNeXtVAE(nn.Module):
    @nn.compact
    def __call__(self, img, cond, training=True):
        z_mean, z_log_var, z = Encoder()(img, cond, training=training)
        reconstructed = Decoder()(z, cond, training=training)
        return reconstructed, z_mean, z_log_var


class ConvNeXtVAEGen(nn.Module):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], 32))
        return Decoder()(z, cond, training=False)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = ConvNeXtVAE(), ConvNeXtVAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-5)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(default_generate_fn(model_gen))
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'convnext_vae', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
