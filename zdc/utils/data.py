import os

import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load(path, scaler, val_size=0.1, test_size=0.2):
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

    r_train, r_test, p_train, p_test = train_test_split(responses, particles, test_size=test_size, shuffle=False)
    r_train, r_val, p_train, p_val = train_test_split(r_train, p_train, test_size=val_size / (1 - test_size), shuffle=False)

    return r_train, r_val, r_test, p_train, p_val, p_test


def batches(*x, batch_size=None, shuffle_key=None):
    assert batch_size is not None

    if shuffle_key is not None:
        perm = jax.random.permutation(shuffle_key, jnp.arange(len(x[0])))
        x = tuple(x_i[perm] for x_i in x)

    n = len(x[0])

    for i in range(0, n, batch_size):
        yield (x_i[i:i + batch_size] for x_i in x)
