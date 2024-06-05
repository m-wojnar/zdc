import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    color_w = plt.colormaps.get_cmap('viridis')(0.25)
    color_m = plt.colormaps.get_cmap('viridis')(0.75)

    df = pd.read_csv('results/diffusion_eta.csv')
    x = df['eta']

    _, ax = plt.subplots()

    ax.plot(x, df['wasserstein'], color=color_w)
    ax.scatter(x, df['wasserstein'], color=color_w, s=10)

    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel('Wasserstein')
    ax.set_ylim(0, 6)

    ax2 = ax.twinx()
    ax2.plot(x, df['mae'], color=color_m)
    ax2.scatter(x, df['mae'], color=color_m, s=10)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0, 30)

    ax2.scatter(None, None, color=color_m, s=10, label='MAE')
    ax2.scatter(None, None, color=color_w, s=10, label='Wasserstein')
    ax2.legend(loc='lower right')

    ax.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('diffusion_eta.pdf', bbox_inches='tight')
    plt.show()
