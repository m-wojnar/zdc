import os

import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DEFAULT_PATH = '/net/pr2/projects/plgrid/plggdaisnet/mwojnar/zdc/data'


def load(path=DEFAULT_PATH, scaler='standard', val_size=0.1, test_size=0.2, load_pdgid=False):
    responses = jnp.load(os.path.join(path, 'data_nonrandom_responses.npz'))['arr_0'].astype(float)
    responses = responses[..., None]
    responses = jnp.log(responses + 1)

    particles = jnp.load(os.path.join(path, 'data_nonrandom_particles.npz'))['arr_0'].astype(float)
    particles, pdgid = particles[..., :-1], particles[..., -1]

    r_train, r_test, p_train, p_test, pdgid_train, pdgid_test = train_test_split(responses, particles, pdgid, test_size=test_size, random_state=42)
    r_train, r_val, p_train, p_val, pdgid_train, pdgid_val = train_test_split(r_train, p_train, pdgid_train, test_size=val_size / (1 - test_size), random_state=43)

    if scaler == 'standard':
        scaler = StandardScaler()
        scaler.fit(p_train)
        p_train, p_val, p_test = map(scaler.transform, (p_train, p_val, p_test))
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(p_train)
        p_train, p_val, p_test = map(scaler.transform, (p_train, p_val, p_test))
    elif scaler != 'none':
        raise ValueError('Unknown scaler')

    if load_pdgid:
        p_train = jnp.concatenate([p_train, pdgid_train[..., None]], axis=-1)
        p_val = jnp.concatenate([p_val, pdgid_val[..., None]], axis=-1)
        p_test = jnp.concatenate([p_test, pdgid_test[..., None]], axis=-1)

    return r_train, r_val, r_test, p_train, p_val, p_test


def get_samples(path=DEFAULT_PATH, scaler='standard', load_pdgid=False):
    responses = jnp.load(os.path.join(path, 'sample_responses.npz'))['arr_0'].astype(float)
    responses = responses[..., None]
    responses = jnp.log(responses + 1)

    particles = jnp.load(os.path.join(path, 'sample_particles.npz'))['arr_0'].astype(float)
    particles, pdgid = particles[..., :-1], particles[..., -1]

    if load_pdgid:
        particles = jnp.concatenate([particles, pdgid[..., None]], axis=-1)

    if scaler != 'standard':
        raise ValueError('Samples were scaled with StandardScaler')

    return responses, particles


def batches(*x, batch_size, shuffle_key=None):
    n = len(x[0])

    if shuffle_key is not None:
        perm = jax.random.permutation(shuffle_key, jnp.arange(n))
        x = tuple(x_i[perm] for x_i in x)

    for i in range(0, n, batch_size):
        yield tuple(x_i[i:i + batch_size] for x_i in x)
