from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import optax
import optuna

from zdc.architectures.vit import Encoder, Decoder
from zdc.models.autoencoder.variational import VAE, loss_fn
from zdc.utils.data import load, batches
from zdc.utils.nn import opt_with_cosine_schedule, init, gradient_step
from zdc.utils.train import default_eval_fn, default_generate_fn


def suggest_optimizer(trial, epochs, batch_size, n_examples):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    beta_1 = trial.suggest_float('beta_1', 0.5, 1.)
    beta_2 = trial.suggest_float('beta_2', 0.5, 1.)
    epsilon = trial.suggest_float('epsilon', 1e-10, 1e-2, log=True)

    use_cosine_decay = trial.suggest_categorical('cosine_decay', [True, False])
    use_nesterov = trial.suggest_categorical('nesterov', [True, False])
    use_weight_decay = trial.suggest_categorical('use_weight_decay', [True, False])

    if use_weight_decay:
        weight_decay = trial.suggest_float('weight_decay', 1e-8, 1., log=True)
        optimizer = partial(optax.adamw, b1=beta_1, b2=beta_2, eps=epsilon, weight_decay=weight_decay, nesterov=use_nesterov)
    else:
        optimizer = partial(optax.adam, b1=beta_1, b2=beta_2, eps=epsilon, nesterov=use_nesterov)

    if use_cosine_decay:
        pct_start = trial.suggest_float('pct_start', 0., 0.5)
        div_factor = trial.suggest_float('div_factor', 10., 1e3, log=True)
        final_div_factor = trial.suggest_float('final_div_factor', 10., 1e4, log=True)
        return opt_with_cosine_schedule(optimizer, learning_rate, pct_start, div_factor, final_div_factor, n_examples, epochs, batch_size)
    else:
        return optimizer(learning_rate)


def objective(trial, train_dataset, val_dataset, epochs=100, batch_size=256, n_examples=214746):
    optimizer = suggest_optimizer(trial, epochs, batch_size, n_examples)

    init_key, train_key, val_key, shuffle_key = jax.random.split(jax.random.PRNGKey(42), 4)

    model = VAE(Encoder, Decoder)
    params, state = init(model, init_key, r_train[:5], p_train[:5])
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(default_generate_fn(model))

    for _ in range(epochs):
        shuffle_key, shuffle_train_subkey, shuffle_val_subkey = jax.random.split(shuffle_key, 3)

        for batch in batches(*train_dataset, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, (state, *losses) = train_fn(params, (state, subkey, *batch), opt_state)

        generated, original = [], []

        for batch in batches(*val_dataset, batch_size=batch_size, shuffle_key=shuffle_val_subkey):
            val_key, subkey = jax.random.split(val_key)
            generated.append(generate_fn(params, state, subkey, *batch))
            original.append(batch[0])

        generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
        _, _, val_wasserstein = default_eval_fn(generated, *original)

    return val_wasserstein


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--database', required=True, type=str)
    args.add_argument('--name', required=True, type=str)
    args.add_argument('--trials', required=False, type=int, default=100)
    args = args.parse_args()

    r_train, r_val, _, p_train, p_val, _ = load()

    study = optuna.create_study(
        storage=args.database,
        study_name=args.name,
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler()
    )

    study.optimize(
        partial(objective, train_dataset=(r_train, p_train), val_dataset=(r_val, p_val)),
        n_trials=args.trials,
        gc_after_trial=True
    )
