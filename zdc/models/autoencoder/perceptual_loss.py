from functools import partial

import flaxmodels as fm
import jax
import optax

from zdc.models.autoencoder.variational import VAEGen, VAE
from zdc.utils.data import load
from zdc.utils.losses import kl_loss, mse_loss, perceptual_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


def loss_fn(params, state, key, img, cond, model, image_processor, perceptual_model, perceptual_weight, kl_weight):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img, cond)
    kl = kl_loss(z_mean, z_log_var)
    mse = mse_loss(img, reconstructed)
    perceptual = perceptual_loss(img, reconstructed, image_processor, perceptual_model)
    loss = perceptual_weight * perceptual + kl_weight * kl + mse
    return loss, (state, loss, perceptual, kl, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    model, model_gen = VAE(), VAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    perceptual_model = fm.VGG16(include_head=False, pretrained='imagenet')
    perceptual_params = perceptual_model.init(key, r_train[:5])
    perceptual_model_fn = lambda x: perceptual_model.apply(perceptual_params, x, train=False)
    perprocess_fn = lambda x: (x / (x.max(axis=(1, 2, 3), keepdims=True) + 1e-6)) + 1e-6

    loss_fn = partial(loss_fn, model=model, image_processor=perprocess_fn, perceptual_model=perceptual_model_fn, perceptual_weight=1.0, kl_weight=0.7)
    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=loss_fn))
    generate_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])
    train_metrics = ('loss', 'perceptual', 'kl', 'mse')

    train_loop(
        'perceptual_loss', train_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
