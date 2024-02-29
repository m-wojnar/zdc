import os

import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load(path, scaler, test_size=0.2):
    responses = jnp.load(os.path.join(path, 'data_nonrandom_responses.npz'))['arr_0'].astype(float)
    responses = responses[..., None]
    responses = jnp.log(responses + 1)

    particles = jnp.load(os.path.join(path, 'data_nonrandom_particles.npz'))['arr_0'].astype(float)

    if scaler == 'standard':
        particles = StandardScaler().fit_transform(particles)
    elif scaler == 'minmax':
        particles = MinMaxScaler().fit_transform(particles)
    elif scaler != 'none':
        raise ValueError('Unknown scaler')

    return train_test_split(responses, particles, test_size=test_size, shuffle=False)


def batches(*x, batch_size=None):
    assert batch_size is not None
    n = len(x[0])

    for i in range(0, n, batch_size):
        yield (x_i[i:i + batch_size] for x_i in x)
