from functools import partial

import jax
import jax.numpy as jnp
import optax

from zdc.models.autoencoder.noise_generator import NGVAE, NGVAEGen
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


def eps_schedule(diameter, blur, scaling):
    return jnp.concatenate([
        jnp.asarray([diameter ** 2]),
        jnp.exp(jnp.arange(2 * jnp.log(diameter), 2 * jnp.log(blur), 2 * jnp.log(scaling))),
        jnp.asarray([blur ** 2])
    ])


def cost(x, y):
    D_xx = (x * x).sum(axis=-1)[:, :, None]
    D_xy = x @ y.transpose(0, 2, 1)
    D_yy = (y * y).sum(axis=-1)[:, None, :]
    return (D_xx - 2 * D_xy + D_yy) / 2


def softmin(eps, C_xy, h_y):
    return -eps * jax.nn.logsumexp(h_y.reshape(1, 1, -1) - C_xy / eps, axis=2)


def sinkhorn_loss(x, y, eps_list):
    a, b = jnp.ones(x.shape[0]) / x.shape[0], jnp.ones(y.shape[0]) / y.shape[0]
    x, y, a, b = jax.tree_map(lambda x: x[None, ...], (x, y, a, b))
    a_log, b_log = jnp.log(a), jnp.log(b)

    C_xy = cost(x, jax.lax.stop_gradient(y))
    C_yx = cost(y, jax.lax.stop_gradient(x))
    C_xx = cost(x, jax.lax.stop_gradient(x))
    C_yy = cost(y, jax.lax.stop_gradient(y))

    def sinkhorn_iter():
        eps = eps_list[0]

        g_ab = softmin(eps, C_yx, a_log)
        f_ba = softmin(eps, C_xy, b_log)
        f_aa = softmin(eps, C_xx, a_log)
        g_bb = softmin(eps, C_yy, b_log)

        for i in range(len(eps_list)):
            eps = eps_list[i]

            ft_ba = softmin(eps, C_xy, b_log + g_ab / eps)
            gt_ab = softmin(eps, C_yx, a_log + f_ba / eps)
            ft_aa = softmin(eps, C_xx, a_log + f_aa / eps)
            gt_bb = softmin(eps, C_yy, b_log + g_bb / eps)

            f_ba, g_ab = (f_ba + ft_ba) / 2, (g_ab + gt_ab) / 2
            f_aa, g_bb = (f_aa + ft_aa) / 2, (g_bb + gt_bb) / 2

        return f_aa, g_bb, g_ab, f_ba

    f_aa, g_bb, g_ab, f_ba = jax.lax.stop_gradient(sinkhorn_iter())
    eps = eps_list[-1]

    f_ba, g_ab = (
        softmin(eps, C_xy, jax.lax.stop_gradient(b_log + g_ab / eps)),
        softmin(eps, C_yx, jax.lax.stop_gradient(a_log + f_ba / eps)),
    )

    f_aa = softmin(eps, C_xx, jax.lax.stop_gradient(a_log + f_aa / eps))
    g_bb = softmin(eps, C_yy, jax.lax.stop_gradient(b_log + g_bb / eps))

    return (f_ba - f_aa).mean() + (g_ab - g_bb).mean()


def loss_fn(params, state, key, img, cond, model, sinkhorn_weight, eps_list):
    (reconstructed, enc, le), state = forward(model, params, state, key, img, cond)
    sinkhorn = sinkhorn_loss(jnp.concatenate([enc, cond], axis=1), jnp.concatenate([le, cond], axis=1), eps_list)
    mse = mse_loss(img, reconstructed)
    loss = sinkhorn_weight * sinkhorn + mse
    return loss, (state, loss, sinkhorn, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = NGVAE(), NGVAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    eps_list = eps_schedule(diameter=1e-2, blur=1e-5, scaling=0.95)
    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, sinkhorn_weight=20., eps_list=eps_list)))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'sinkhorn', 'mse')

    train_loop(
        'sinkhorn', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
