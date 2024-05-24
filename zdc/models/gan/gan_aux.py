from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.gan.aux_regressor import AuxRegressor
from zdc.models.gan.gan import GAN, disc_optimizer, gen_optimizer, step_fn, disc_loss_fn
from zdc.utils.data import load
from zdc.utils.losses import xentropy_loss, mse_loss
from zdc.utils.nn import init, forward, get_layers, load_model
from zdc.utils.train import train_loop, default_generate_fn


def gen_loss_fn(gen_params, disc_params, state, forward_key, img, cond, rand_cond, model, aux, aux_weight=1.0):
    (generated, _, fake_output), state = forward(model, gen_params | disc_params, state, forward_key, img, cond, rand_cond)

    aux_key_img, aux_key_gen = jax.random.split(forward_key)
    center_true, _ = forward(*aux, aux_key_img, img, cond, None, method='gen')
    center_pred, _ = forward(*aux, aux_key_gen, generated, cond, None, method='gen')
    x_true, y_true = center_true[:, 0], center_true[:, 1]
    x_pred, y_pred = center_pred[:, 0], center_pred[:, 1]

    adv_loss = xentropy_loss(fake_output, jnp.ones_like(fake_output))
    aux_loss = mse_loss(x_true, x_pred) + mse_loss(y_true, y_pred)
    loss = adv_loss + aux_weight * aux_loss

    gen_acc = (fake_output > 0).mean()
    return loss, (state, loss, gen_acc)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    aux_model = AuxRegressor()
    aux_params, aux_state = load_model('checkpoints/aux_regressor/epoch_100.pkl.lz4')

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = GAN()
    params, state = init(model, init_key, r_train[:5], p_train[:5], p_train[:5], print_summary=True)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, model=model),
        gen_loss_fn=partial(gen_loss_fn, model=model, aux=(aux_model, aux_params, aux_state))
    ))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('disc_loss', 'gen_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc')

    train_loop(
        'gan_aux', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key
    )
