from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.gan.gan import GAN, disc_optimizer, gen_optimizer, gen_loss_fn, step_fn
from zdc.utils.data import load
from zdc.utils.losses import xentropy_loss
from zdc.utils.nn import init, forward, get_layers
from zdc.utils.train import train_loop, default_generate_fn


gamma = 0.1


def disc_regularizer(disc_params, state, key, img, cond, real_logits, fake_logits, generated, model):
    def vgrad(f, x):
        y, vjp_fn = jax.vjp(f, x)
        return vjp_fn(jnp.ones_like(y))[0]

    def grad_norm(data):
        data = jax.lax.stop_gradient(data)
        grad = vgrad(lambda x: forward(model, disc_params, state, key, x, cond, method='disc')[0], data)
        grad_norm = jnp.linalg.norm(jax.lax.collapse(grad, 1), axis=1, keepdims=True)
        return grad_norm

    real_sigmoid, fake_sigmoid = jax.nn.sigmoid(real_logits), jax.nn.sigmoid(fake_logits)
    real_grad_norm, fake_grad_norm = grad_norm(img), grad_norm(generated)

    real_reg = jnp.multiply((1.0 - real_sigmoid) ** 2, real_grad_norm ** 2)
    fake_reg = jnp.multiply(fake_sigmoid ** 2, fake_grad_norm ** 2)

    return jnp.mean(real_reg + fake_reg)


def disc_loss_fn(disc_params, gen_params, state, forward_key, img, cond, rand_cond, model):
    (generated, real_output, fake_output), state = forward(model, disc_params | gen_params, state, forward_key, img, cond, rand_cond)

    real_loss = xentropy_loss(real_output, jnp.ones_like(real_output))
    fake_loss = xentropy_loss(fake_output, jnp.zeros_like(fake_output))
    reg_loss = disc_regularizer(disc_params, state, forward_key, img, cond, real_output, fake_output, generated, model)
    loss = real_loss + fake_loss + gamma / 2 * reg_loss

    disc_real_acc = (real_output > 0).mean()
    disc_fake_acc = (fake_output < 0).mean()
    return loss, (state, loss, disc_real_acc, disc_fake_acc)


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
        'gan_gp_annealed', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key
    )
