from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.gan.gan import GAN, disc_loss_fn, step_fn, disc_optimizer, gen_optimizer
from zdc.utils.data import load
from zdc.utils.losses import xentropy_loss, mse_loss
from zdc.utils.nn import init, forward, get_layers
from zdc.utils.train import train_loop, default_generate_fn


def gen_loss_fn(gen_params, disc_params, state, forward_key, img, cond, rand_cond, model, l2_weight=0.001):
    (generated, _, fake_output), state = forward(model, gen_params | disc_params, state, forward_key, img, cond, rand_cond)

    adv_loss = xentropy_loss(fake_output, jnp.ones_like(fake_output))
    l2_loss = mse_loss(generated, img)
    loss = adv_loss + l2_weight * l2_loss

    gen_acc = (fake_output > 0).mean()
    return loss, (state, loss, gen_acc)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

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
        gen_loss_fn=partial(gen_loss_fn, model=model)
    ))
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('disc_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_loss', 'gen_acc')

    train_loop(
        'gan_l2', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key
    )
