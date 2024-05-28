import jax
import jax.numpy as jnp

from zdc.architectures.vit import Encoder, Decoder
from zdc.models.autoencoder.variational import VAE
from zdc.utils.data import load, batches
from zdc.utils.nn import load_model
from zdc.utils.train import default_generate_fn
from zdc.utils.wasserstein import sum_channels_parallel


if __name__ == '__main__':
    test_key = jax.random.PRNGKey(42)

    batch_size, n_rep = 256, 5
    _, _, r_test, _, _, p_test = load()

    model = VAE(Encoder, Decoder)
    params, state = load_model('../models/autoencoder/checkpoints/variational/epoch_100.pkl.lz4')
    generate_fn = jax.jit(default_generate_fn(model))

    channels = []

    for batch in batches(r_test, p_test, batch_size=batch_size):
        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            channels.append(sum_channels_parallel(generate_fn(params, state, subkey, *batch)))

    channels = jnp.concatenate(channels, axis=0)

    jnp.savez('autoencoder.npz', channels)
