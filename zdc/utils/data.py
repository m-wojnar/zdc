import os

import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load(path, scaler, val_size=0.1, test_size=0.2, load_pdgid=False):
    responses = jnp.load(os.path.join(path, 'data_nonrandom_responses.npz'))['arr_0'].astype(float)
    responses = responses[..., None]
    responses = jnp.log(responses + 1)

    particles = jnp.load(os.path.join(path, 'data_nonrandom_particles.npz'))['arr_0'].astype(float)
    particles, pdgid = particles[..., :-1], particles[..., -1]

    if scaler == 'standard':
        particles = StandardScaler().fit_transform(particles)
    elif scaler == 'minmax':
        particles = MinMaxScaler().fit_transform(particles)
    elif scaler != 'none':
        raise ValueError('Unknown scaler')

    if load_pdgid:
        particles = jnp.concatenate([particles, pdgid[..., None]], axis=-1)

    r_train, r_test, p_train, p_test = train_test_split(responses, particles, test_size=test_size, shuffle=False)
    r_train, r_val, p_train, p_val = train_test_split(r_train, p_train, test_size=val_size / (1 - test_size), shuffle=False)

    return r_train, r_val, r_test, p_train, p_val, p_test


def get_samples(*xs):
    return jax.tree_map(lambda x: x[50:100:4], xs)


def batches(*x, batch_size, shuffle_key=None):
    n = len(x[0])

    if shuffle_key is not None:
        perm = jax.random.permutation(shuffle_key, jnp.arange(n))
        x = tuple(x_i[perm] for x_i in x)

    for i in range(0, n, batch_size):
        yield tuple(x_i[i:i + batch_size] for x_i in x)
