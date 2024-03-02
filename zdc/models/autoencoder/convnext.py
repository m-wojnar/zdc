import os
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from tqdm import trange

from zdc.layers import Concatenate, ConvNeXtV2Embedding, ConvNeXtV2Stage, GlobalAveragePooling, Reshape, Sampling, UpSample
from zdc.models.autoencoder.variational import eval_fn, loss_fn
from zdc.utils.data import load, batches
from zdc.utils.metrics import Metrics
from zdc.utils.nn import init, forward, gradient_step, save_model, print_model


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
        x = nn.LayerNorm(epsilon=1e-6)(x)
        x = Concatenate()(x, cond)
        x = nn.Dense(2 * self.latent_dim)(x)
        x = nn.gelu(x)

        z_mean = nn.Dense(self.latent_dim)(x)
        z_log_var = nn.Dense(self.latent_dim)(x)
        z = Sampling()(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(nn.Module):
    kernel_size: int = 3
    max_drop_rate: float = 0.33
    depths: tuple = (1, 3, 1, 1)
    projection_dims: tuple = (192, 96, 48, 24)
    drop_rates = [r.tolist() for r in jnp.split(jnp.linspace(max_drop_rate, 0., sum(depths)), jnp.cumsum(jnp.array(depths))[:-1])]

    @nn.compact
    def __call__(self, z, cond, training=True):
        x = Concatenate()(z, cond)
        x = nn.Dense(3 * 3 * self.projection_dims[0])(x)
        x = nn.gelu(x)
        x = Reshape((-1, 3, 3, self.projection_dims[0]))(x)
        x = nn.LayerNorm(epsilon=1e-6)(x)

        for projection_dim, drop_rates in zip(self.projection_dims, self.drop_rates):
            x = UpSample()(x)
            x = ConvNeXtV2Stage(1, projection_dim, self.kernel_size, drop_rates)(x, training=training)

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
    batch_size = 128
    kl_weight = 0.7
    n_reps = 5
    lr = 4e-4
    epochs = 100
    seed = 42

    key = jax.random.PRNGKey(seed)
    init_key, train_key, val_key, test_key, plot_key = jax.random.split(key, 5)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    model, model_gen = ConvNeXtVAE(), ConvNeXtVAEGen()
    params, state = init(model, init_key, r_sample, p_sample)
    print_model(params)

    optimizer = optax.rmsprop(lr, decay=0.8, momentum=0.1)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=kl_weight)))
    eval_fn = jax.jit(partial(eval_fn, model=model, kl_weight=kl_weight, n_reps=n_reps))
    eval_metrics = ('loss', 'kl', 'mse', 'mae', 'wasserstein')

    metrics = Metrics(job_type='train', name='vae_convnext')
    os.makedirs('checkpoints/vae_convnext', exist_ok=True)

    for epoch in trange(epochs, desc='Epochs'):
        for batch in batches(r_train, p_train, batch_size=batch_size):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, loss, (state, kl, mse) = train_fn(params, (state, subkey, *batch), opt_state)
            metrics.add({'loss': loss, 'kl': kl, 'mse': mse}, 'train')

        metrics.log(epoch)

        for batch in batches(r_val, p_val, batch_size=batch_size):
            val_key, subkey = jax.random.split(val_key)
            metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'val')

        metrics.log(epoch)

        plot_key, subkey = jax.random.split(plot_key)
        metrics.plot_responses(r_sample, forward(model_gen, params, state, subkey, p_sample)[0], epoch)

        save_model(params, state, f'checkpoints/convnext/epoch_{epoch + 1}.pkl.lz4')

    for batch in batches(r_test, p_test, batch_size=batch_size):
        test_key, subkey = jax.random.split(test_key)
        metrics.add(dict(zip(eval_metrics, eval_fn(params, state, subkey, *batch))), 'test')

    metrics.log(epochs)
