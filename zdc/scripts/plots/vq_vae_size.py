import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    color_w = plt.colormaps.get_cmap('viridis')(0.25)
    color_m = plt.colormaps.get_cmap('viridis')(0.75)

    df = pd.read_csv('vq_vae_size.csv')
    x = df['size'] * 1e6

    _, ax = plt.subplots()

    plt.plot(x, df['wasserstein'], color=color_w)
    plt.scatter(x, df['wasserstein'], color=color_w, s=10, label='Wasserstein')

    plt.plot(x, df['mae'], color=color_m)
    plt.scatter(x, df['mae'], color=color_m, s=10, label='MAE')

    plt.xlabel('Number of parameters')
    plt.ylabel('Metric value')
    plt.xscale('log')
    plt.yticks(range(0, 16, 3))
    plt.xlim(1e5, 1e8)
    plt.ylim(0, 15)
    plt.legend()

    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('vq_vae_size.pdf', bbox_inches='tight')
    plt.show()
