import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    color = plt.colormaps.get_cmap('viridis')(0.25)

    df = pd.read_csv('vq_vae_size.csv')

    _, ax = plt.subplots()

    plt.plot(df['size'] * 1e6, df['wasserstein'], linestyle='--', color=color)
    plt.scatter(df['size'] * 1e6, df['wasserstein'], color=color, s=df['size'] * 20)

    plt.xlabel('Number of parameters')
    plt.ylabel('Wasserstein metric')
    plt.xscale('log')
    plt.yticks(range(0, 16, 3))
    plt.xlim(1e5, 1e8)
    plt.ylim(0, 15)

    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('vq_vae_size.pdf', bbox_inches='tight')
    plt.show()
