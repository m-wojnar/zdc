from functools import partial

import jax
import optax

from zdc.models.autoencoder.variational import loss_fn, eval_fn
from zdc.models.autoencoder.vit import ViTVAE, ViTVAEGen
from zdc.utils.data import load
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard')
    r_sample, p_sample = jax.tree_map(lambda x: x[20:30], (r_train, p_train))

    vit_config = dict(latent_dim=128, num_heads=8, drop_rate=0.2, embedding_dim=192, depth=6)
    model, model_gen = ViTVAE(**vit_config), ViTVAEGen(**vit_config)
    params, state = init(model, init_key, r_sample, p_sample, print_summary=True)

    train_steps = 100 * len(r_train) // 128
    lr = optax.cosine_onecycle_schedule(train_steps, peak_value=2e-4, pct_start=0.1, div_factor=20, final_div_factor=100)
    optimizer = optax.adamw(lr, weight_decay=1e-2)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    eval_fn = jax.jit(partial(eval_fn, model=model, kl_weight=0.7))
    plot_fn = jax.jit(lambda *x: forward(model_gen, *x)[0])

    train_metrics = ('loss', 'kl', 'mse')
    eval_metrics = ('loss', 'kl', 'mse', 'mae', 'wasserstein')

    train_loop(
        'vit_big', train_fn, eval_fn, plot_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test), r_sample, p_sample,
        train_metrics, eval_metrics, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
