import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH
from zdc.utils.data import load


if __name__ == '__main__':
    r_train, r_val, r_test, p_train, p_val, p_test = load(load_pdgid=True)
    cols = [r'$E$', r'$v_x$', r'$v_y$', r'$v_z$', r'$p_x$', r'$p_y$', r'$p_z$', r'$m$', r'$c$', 'PDGID']

    particles = jnp.concat([p_train, p_val, p_test])
    responses = jnp.concat([r_train, r_val, r_test])
    diversity = jnp.empty(particles.shape[0], dtype=float)

    for u in jnp.unique(particles, axis=0):
        mask = jnp.all(particles == u, axis=1)
        diversity = diversity.at[mask].set(jnp.std(responses[mask], axis=0).sum())

    div_min, div_max = jnp.min(diversity), jnp.max(diversity)
    diversity = (diversity - div_min) / (div_max - div_min)

    rf = RandomForestRegressor(n_estimators=500, random_state=42)
    rf.fit(particles, diversity)

    for col, imp in zip(cols, rf.feature_importances_):
        print(f'{col}: {imp:.2f}')

    particles = particles.at[:, -1].set(diversity)
    cols[-1] = r'$f_{div}$'
    df = pd.DataFrame(data=particles, columns=cols)

    plt.rcParams.update(PLOT_PARAMS)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.7 * COLUMN_WIDTH))
    im = ax.imshow(df.corr(), cmap='viridis')
    ax.set_xticks(range(len(cols)), labels=cols)
    ax.set_yticks(range(len(cols)), labels=cols)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('dataset_correlation.pdf', bbox_inches='tight')
    plt.show()
