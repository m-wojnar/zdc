import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HIGHT


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv('latent_space.csv')
    latent_size = df['latent_size']
    wasserstein = df['wasserstein']

    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0.1, 0.9, len(latent_size)))
    _, ax = plt.subplots(figsize=(0.8 * COLUMN_WIDTH, 0.8 * COLUMN_HIGHT))

    plt.plot(latent_size, wasserstein, color='gray', linestyle='--', zorder=0)

    for size, was, color in zip(latent_size, wasserstein, colors):
        ax.scatter(size, was, color=color, label=size, zorder=1, s=15)

    plt.xlabel('Latent space size')
    plt.ylabel('Wasserstein')
    plt.yticks(range(0, 16, 3))
    plt.xscale('log')
    plt.ylim(0, 15)

    plt.legend(title='Latent space size', loc='lower center', ncol=4)
    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('latent_size.pdf', bbox_inches='tight')
    plt.show()
