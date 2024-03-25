from functools import partial

import jax
import jax.numpy as jnp
import optax

from zdc.utils.wasserstein import wasserstein_channels


def kl_loss(mean, log_var):
    return -0.5 * (1. + log_var - jnp.square(mean) - jnp.exp(log_var)).sum(axis=1).mean()


def mse_loss(x, y):
    return jnp.square(x - y).reshape(x.shape[0], -1).sum(axis=-1).mean()


def mae_loss(x, y):
    return jnp.abs(x - y).reshape(x.shape[0], -1).sum(axis=-1).mean()


def wasserstein_loss(ch_true, ch_pred):
    return wasserstein_channels(ch_true, ch_pred).mean()


def xentropy_loss(x, y):
    return optax.sigmoid_binary_cross_entropy(x, y).reshape(x.shape[0], -1).sum(axis=-1).mean()


def perceptual_loss(img, reconstructed, image_processor, perceptual_model):
    img = perceptual_model(image_processor(img))
    reconstructed = perceptual_model(image_processor(reconstructed))
    return mse_loss(img, reconstructed)


def sinkhorn_loss(diameter, blur, scaling):
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

    def sinkhorn(x, y, eps_list):
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

    return partial(sinkhorn, eps_list=eps_schedule(diameter, blur, scaling))
