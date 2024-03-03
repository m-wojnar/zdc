import jax
import lz4.frame
import optax
from cloudpickle import cloudpickle
from clu.parameter_overview import get_parameter_overview


def gradient_step(params, loss_params, opt_state, optimizer, loss_fn):
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, *loss_params)
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, aux


def init(model, key, *x):
    params_key, zdc_key, dropout_key = jax.random.split(key, 3)

    variables = model.init({'params': params_key, 'zdc': zdc_key, 'dropout': dropout_key}, *x)
    params = variables.pop('params')
    state = variables

    return params, state


def forward(model, params, state, key, *x):
    zdc_key, dropout_key = jax.random.split(key)
    return model.apply({'params': params, **state}, *x, rngs={'zdc': zdc_key, 'dropout': dropout_key}, mutable=list(state.keys()))


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


def print_model(params):
    print(get_parameter_overview(params, include_stats=False))
