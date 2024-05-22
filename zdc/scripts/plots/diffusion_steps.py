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

    ax.plot(df['time'] / n, df['wasserstein'], linestyle='--', marker='o', color=color_w, label='Wasserstein')
    ax.plot(df['time'] / n, df['mae'], linestyle='--', marker='o', color=color_m, label='MAE')

    ax.set_xlabel('Time per sample [s]')
    ax.set_ylabel('Metric value')
    ax.set_xscale('log')
    ax.set_ylim(0, 30)
    ax.legend()

    ax2 = ax.twiny()
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel('Number of steps')
    ax2.set_xticks(df['time'] / n, labels=df['steps'])
    ax2.minorticks_off()

    ax.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('diffusion_steps.pdf', bbox_inches='tight')
    plt.show()
