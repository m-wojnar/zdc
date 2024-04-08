from functools import partial

import jax
import optax

from zdc.models.autoencoder.variational import loss_fn
from zdc.models.autoencoder.vit import ViTVAE, ViTVAEGen
from zdc.utils.data import load
from zdc.utils.nn import init, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop, default_generate_fn


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')

    vit_config = dict(latent_dim=128, num_heads=6, drop_rate=0.1, embedding_dim=288, depth=6)
    model, model_gen = ViTVAE(**vit_config), ViTVAEGen(**vit_config)
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 1e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(default_generate_fn(model_gen))
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'vit_big', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
