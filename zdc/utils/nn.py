import jax
import lz4.frame
import optax
from cloudpickle import cloudpickle
from clu.parameter_overview import get_parameter_overview


def gradient_step(params, loss_params, opt_state, optimizer, loss_fn):
    grads, aux = jax.grad(loss_fn, has_aux=True)(params, *loss_params)
    updates, opt_state = optimizer.update(grads, opt_state, params=params, grad_fn=jax.grad(lambda p, _: loss_fn(p, *loss_params)[0]))
    params = optax.apply_updates(params, updates)

    return params, opt_state, aux


def init(model, key, *x, print_summary=False):
    params_key, zdc_key, dropout_key = jax.random.split(key, 3)

    variables = model.init({'params': params_key, 'zdc': zdc_key, 'dropout': dropout_key}, *x)
    params = variables.pop('params')
    state = variables

    if print_summary:
        print(get_parameter_overview(params, include_stats=False))

    return params, state


def forward(model, params, state, key, *x, method=None):
    zdc_key, dropout_key = jax.random.split(key)
    return model.apply({'params': params, **state}, *x, rngs={'zdc': zdc_key, 'dropout': dropout_key}, mutable=list(state.keys()), method=method)


def get_layers(params, layer_names):
    if isinstance(layer_names, str):
        return {layer_names: params[layer_names]}
    else:
        return {name: params[name] for name in layer_names}


def save_model(params, state, path):
    with lz4.frame.open(path, 'wb') as f:
        cloudpickle.dump((params, state), f)


def load_model(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.load(f)


def opt_with_cosine_schedule(optimizer, peak_value, pct_start=0.1, div_factor=25, final_div_factor=100, n_examples=214746, epochs=100, batch_size=256):
    train_steps = epochs * n_examples // batch_size
    lr = optax.cosine_onecycle_schedule(train_steps, peak_value=peak_value, pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor)
    return optimizer(lr)
