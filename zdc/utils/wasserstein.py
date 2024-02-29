import jax.numpy as jnp

from zdc.models import INPUT_SHAPE


coords = jnp.ogrid[0:INPUT_SHAPE[0], 0:INPUT_SHAPE[1]]
half_x = INPUT_SHAPE[0] // 2
half_y = INPUT_SHAPE[1] // 2

checkerboard = (coords[0] + coords[1]) % 2 != 0
checkerboard = checkerboard.reshape(1, checkerboard.shape[0], checkerboard.shape[1])
checkerboard = checkerboard.astype(float)

mask5 = jnp.copy(checkerboard)

checkerboard = (coords[0] + coords[1]) % 2 == 0
checkerboard = checkerboard.reshape(1, checkerboard.shape[0], checkerboard.shape[1])

mask1 = jnp.zeros((1, INPUT_SHAPE[0], INPUT_SHAPE[1]))
mask1 = mask1.at[:, :half_x, :half_y].set(checkerboard[:, :half_x, :half_y])

mask2 = jnp.zeros((1, INPUT_SHAPE[0], INPUT_SHAPE[1]))
mask2 = mask2.at[:, :half_x, half_y:].set(checkerboard[:, :half_x, half_y:])

mask3 = jnp.zeros((1, INPUT_SHAPE[0], INPUT_SHAPE[1]))
mask3 = mask3.at[:, half_x:, :half_y].set(checkerboard[:, half_x:, :half_y])

mask4 = jnp.zeros((1, INPUT_SHAPE[0], INPUT_SHAPE[1]))
mask4 = mask4.at[:, half_x:, half_y:].set(checkerboard[:, half_x:, half_y:])

masks = [mask1, mask2, mask3, mask4, mask5]


def sum_channels_parallel(data):
    apply_mask = lambda mask: (data * mask).sum(axis=1).sum(axis=1)
    return jnp.stack(list(map(apply_mask, masks)), axis=1)


def wasserstein_distance(x, y):
    x = jnp.sort(x)
    y = jnp.sort(y)

    all_values = jnp.concatenate([x, y], axis=0)
    all_values = jnp.sort(all_values)

    deltas = jnp.diff(all_values)

    x_cdf_indices = jnp.searchsorted(x, all_values[:-1], side='right')
    y_cdf_indices = jnp.searchsorted(y, all_values[:-1], side='right')

    x_cdf = x_cdf_indices.astype(float) / jnp.size(x)
    y_cdf = y_cdf_indices.astype(float) / jnp.size(y)

    return jnp.sum(jnp.multiply(jnp.abs(x_cdf - y_cdf), deltas))


def wasserstein_channels(response_true, response_pred):
    response_true = (jnp.exp(response_true) - 1).reshape((-1, INPUT_SHAPE[0], INPUT_SHAPE[1]))
    response_pred = (jnp.exp(response_pred) - 1).reshape((-1, INPUT_SHAPE[0], INPUT_SHAPE[1]))

    ch_true = sum_channels_parallel(response_true)
    ch_pred = sum_channels_parallel(response_pred)

    return jnp.stack([wasserstein_distance(ch_true[:, i], ch_pred[:, i]) for i in range(5)])


def wasserstein_loss(response_true, response_pred):
    return wasserstein_channels(response_true, response_pred).mean()
