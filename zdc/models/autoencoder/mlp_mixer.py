from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.layers import Concatenate, Flatten, MixerBlock, Patches, PatchEncoder, PatchExpand, Reshape, Sampling
from zdc.models.autoencoder.variational import loss_fn, eval_fn
from zdc.utils.data import get_samples, load
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


class Encoder(nn.Module):
    latent_dim: int
    hidden_dim: int
    tokens_dim: int
    channels_dim: int
    depth: int

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = Patches(patch_size=4)(img)
        x = PatchEncoder(x.shape[1], self.hidden_dim)(x)

        c = nn.Dense(self.hidden_dim)(cond)
        c = Reshape((1, self.hidden_dim))(c)
        x = Concatenate(axis=1)(c, x)

        for _ in range(self.depth):
            x = MixerBlock(self.tokens_dim, self.channels_dim)(x)

        x = Flatten()(x)
        x = nn.Dense(2 * self.latent_dim)(x)
        x = nn.relu(x)
        z_mean = nn.Dense(self.latent_dim)(x)
        z_log_var = nn.Dense(self.latent_dim)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    hidden_dim: int
    tokens_dim: int
    channels_dim: int
    depth: int

    @nn.compact
    def __call__(self, z, cond, training=True):
        x = nn.Dense(11 * 11)(z)
        x = Reshape((11 * 11, 1))(x)
        x = PatchEncoder(11 * 11, self.hidden_dim, positional_encoding=True)(x)

        c = nn.Dense(self.hidden_dim)(cond)
        c = Reshape((1, self.hidden_dim))(c)
        x = Concatenate(axis=1)(c, x)

        for _ in range(self.depth):
            x = MixerBlock(self.tokens_dim, self.channels_dim)(x)

        x = x[:, 1:, :]
        x = PatchExpand(h=11, w=11)(x)
        x = PatchExpand(h=22, w=22)(x)
        x = Reshape((44, 44, self.hidden_dim // 4))(x)
        x = nn.Dense(1)(x)
        x = nn.relu(x)
        return x


class MLPMixerVAE(nn.Module):
    latent_dim: int = 10
    hidden_dim: int = 128
    tokens_dim: int = 64
    channels_dim: int = 512
    depth: int = 6

    @nn.compact
    def __call__(self, img, cond, training=True):
        z_mean, z_log_var, z = Encoder(self.latent_dim, self.hidden_dim, self.tokens_dim, self.channels_dim, self.depth)(img, cond, training=training)
        reconstructed = Decoder(self.hidden_dim, self.tokens_dim, self.channels_dim, self.depth)(z, cond, training=training)
        return reconstructed, z_mean, z_log_var


class MLPMixerVAEGen(MLPMixerVAE):
    @nn.compact
    def __call__(self, cond):
        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], self.latent_dim))
        return Decoder(self.tokens_dim, self.channels_dim, self.depth)(z, cond, training=False)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = get_samples(r_train, p_train)

    model, model_gen = MLPMixerVAE(), MLPMixerVAEGen()
    params, state = init(model, init_key, r_sample, p_sample, print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 1e-3)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    eval_fn = jax.jit(partial(eval_fn, model=model, kl_weight=0.7))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('loss', 'kl', 'mse')
    eval_metrics = ('loss', 'kl', 'mse', 'mae', 'wasserstein')

    train_loop(
        'mlp_mixer', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
