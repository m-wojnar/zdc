from functools import partial

import jax
import jax.numpy as jnp
import optax

from zdc.models.autoencoder.variational import VAE, VAEGen, loss_fn
from zdc.utils.data import load
from zdc.utils.nn import init, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop, default_generate_fn


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load('../../../data', 'standard', load_pdgid=True)
    ps = (p_train, p_val, p_test)

    pdgids = jax.tree_map(lambda x: x[..., -1], ps)
    pdgids = jnp.concatenate(pdgids, axis=0)
    pdgids = jnp.unique(pdgids)
    pdgids_map = {pid.item(): cat for pid, cat in zip(pdgids, range(len(pdgids)))}

    categorical = jax.tree_map(lambda x: jnp.array(jax.tree_map(pdgids_map.get, x[..., -1].tolist())), ps)
    one_hot = jax.tree_map(lambda x: jax.nn.one_hot(x, len(pdgids)), categorical)
    p_train, p_val, p_test = jax.tree_map(lambda x, y: jnp.concatenate([x[..., :-1], y], axis=-1), ps, one_hot)

    model, model_gen = VAE(), VAEGen()
    params, state = init(model, init_key, r_train[:5], p_train[:5], print_summary=True)

    optimizer = opt_with_cosine_schedule(optax.adam, 3e-4)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model, kl_weight=0.7)))
    generate_fn = jax.jit(default_generate_fn(model_gen))
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'one_hot', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key, epochs=100, batch_size=128
    )
