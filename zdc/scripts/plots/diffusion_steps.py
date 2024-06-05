import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH
from zdc.utils.data import load


if __name__ == '__main__':
    *_, p_test = load()
    n = 5 * len(p_test)

    plt.rcParams.update(PLOT_PARAMS)
    color_w = plt.colormaps.get_cmap('viridis')(0.25)
    color_m = plt.colormaps.get_cmap('viridis')(0.75)

    df = pd.read_csv('diffusion_steps.csv')

    _, ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.7 * COLUMN_WIDTH))

    ax.plot(df['time'] / n, df['wasserstein'], marker='o', markersize=4, color=color_w)

    ax.set_xlabel('Time per sample [s]')
    ax.set_ylabel('Wasserstein')
    ax.set_xscale('log')
    ax.set_ylim(0, 12)

    ax2 = ax.twinx()
    ax2.plot(df['time'] / n, df['mae'], marker='o', markersize=4, color=color_m)
    ax2.scatter(None, None, color=color_m, label='MAE', s=10)
    ax2.scatter(None, None, color=color_w, label='Wasserstein', s=10)
    ax2.set_ylabel('MAE')
    ax2.set_ylim(0, 30)
    ax2.legend(loc='lower left')

    ax3 = ax.twiny()
    ax3.set_xscale('log')
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xlabel('Number of steps')
    ax3.set_xticks(df['time'] / n, labels=df['steps'])
    ax3.minorticks_off()

    ax.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('diffusion_steps.pdf', bbox_inches='tight')
    plt.show()
