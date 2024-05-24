from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from scipy.ndimage import center_of_mass

from zdc.architectures.vit import Encoder
from zdc.layers import Flatten
from zdc.models.gan.gan import disc_optimizer
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop


class AuxRegressor(nn.Module):
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    drop_rate: float = 0.1

    @nn.compact
    def __call__(self, img, cond, training=True):
        x = Encoder(self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate)(img, cond, training=training)
        x = nn.Dense(128)(x)
        x = Flatten()(x)
        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(64)(x)
        x = nn.gelu(x)
        x = nn.Dense(2)(x)
        x = jnp.minimum(jnp.maximum(x, 0.), 44.)
        return x

    def gen(self, img, cond):
        return self(img, cond, training=False)


def eval_fn(pred, *dataset):
    _, _, center = dataset
    x_pred, y_pred = pred[:, 0], pred[:, 1]
    x_true, y_true = center[:, 0], center[:, 1]

    loss = mse_loss(x_true, x_pred) + mse_loss(y_true, y_pred)
    return (loss,)


def loss_fn(params, state, forward_key, img, cond, center, model):
    pred, state = forward(model, params, state, forward_key, img, cond)
    loss = eval_fn(pred, img, cond, center)[0]
    return loss, (state, loss)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()
    c_train, c_val, c_test = jax.tree_map(
        lambda x: jnp.array([center_of_mass(np.array(x[i, ..., 0])) for i in range(x.shape[0])]),
        (r_train, r_val, r_test)
    )

    model = AuxRegressor()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)
    opt_state = disc_optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=disc_optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, *x, method='gen')[0])
    metrics = ('loss',)

    train_loop(
        'aux_regressor', train_fn, eval_fn, generate_fn, (r_train, p_train, c_train), (r_val, p_val, c_val), (r_test, p_test, c_test),
        metrics, metrics, params, state, opt_state, train_key
    )
