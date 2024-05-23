from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.gan.gan import GAN, disc_optimizer, gen_optimizer, step_fn
from zdc.utils.data import load
from zdc.utils.nn import init, forward, get_layers
from zdc.utils.train import train_loop, default_generate_fn


def disc_loss_fn(disc_params, gen_params, state, forward_key, img, cond, rand_cond, model):
    def vgrad(f, x):
        y, vjp_fn = jax.vjp(f, x)
        return vjp_fn(jnp.ones_like(y))[0]

    (generated, real_output, fake_output), state = forward(model, disc_params | gen_params, state, forward_key, img, cond, rand_cond)

    real_loss = -jnp.mean(real_output)
    fake_loss = jnp.mean(fake_output)

    grad = vgrad(lambda x: forward(model, disc_params, state, key, x, cond, method='disc')[0], generated)
    gp = 1 - jnp.linalg.norm(jax.lax.collapse(grad, 1), axis=1)
    gp = jnp.mean(gp ** 2)

    loss = real_loss + fake_loss + 10 * gp

    disc_real_acc = (real_output > 0).mean()
    disc_fake_acc = (fake_output < 0).mean()
    return loss, (state, loss, disc_real_acc, disc_fake_acc)


def gen_loss_fn(gen_params, disc_params, state, forward_key, img, cond, rand_cond, model):
    (_, _, fake_output), state = forward(model, gen_params | disc_params, state, forward_key, img, cond, rand_cond)

    loss = -jnp.mean(fake_output)

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
    train_metrics = ('disc_loss', 'gen_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_acc')

    train_loop(
        'wgan', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key
    )
