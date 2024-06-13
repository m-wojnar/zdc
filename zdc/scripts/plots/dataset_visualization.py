import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH
from zdc.utils.data import load


if __name__ == '__main__':
    *_, p_train, p_val, p_test = load(load_pdgid=True)
    particles = jnp.concat([p_train, p_val, p_test])
    particles = jnp.unique(particles, axis=0)

    particles, pdgids = particles[:, :-1], particles[:, -1].astype(int)
    pdgids = LabelEncoder().fit_transform(pdgids)

    embedding = TSNE(perplexity=30.0, random_state=42).fit_transform(particles)

    plt.rcParams.update(PLOT_PARAMS)
    plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=pdgids, cmap='viridis')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('dataset_visualization_30_tsne.png', bbox_inches='tight', dpi=400)
    plt.show()

    color = plt.colormaps.get_cmap('viridis')(0.25)
    features = [
        r'$E \left[ \text{GeV}\right] $', r'$v_x \left[ \text{m} \right]$', r'$v_y \left[ \text{m} \right]$', r'$v_z \left[ \text{m} \right]$',
        r'$p_x \left[ \frac{\text{GeV}}{c}\right] $', r'$p_y \left[ \frac{\text{GeV}}{c}\right] $', r'$p_z \left[ \frac{\text{GeV}}{c}\right] $',
        r'$m \left[ \frac{\text{MeV}}{c^2}\right] $', r'$c \left[ e \right] $'
    ]

    *_, p_train, p_val, p_test = load(scaler='none')
    particles = jnp.concat([p_train, p_val, p_test])
    particles = jnp.unique(particles, axis=0)

    _, axs = plt.subplots(3, 3, figsize=(1.5 * COLUMN_WIDTH, 1.5 * COLUMN_WIDTH))

    for i, (ax, feature) in enumerate(zip(axs.flat, features)):
        ax.hist(particles[:, i], bins=100, color=color)
        ax.set_xlabel(feature)
        ax.grid(True)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('dataset_histograms.pdf', bbox_inches='tight')
    plt.show()
