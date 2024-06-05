import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HIGHT


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    color_w = plt.colormaps.get_cmap('viridis')(0.25)
    color_m = plt.colormaps.get_cmap('viridis')(0.75)

    df = pd.read_csv('transformer.csv')
    x = df['value']

    _, ax = plt.subplots(figsize=(0.8 * COLUMN_WIDTH, 0.8 * COLUMN_HIGHT))

    ax.plot(x, df['wasserstein'], color=color_w)
    ax.scatter(x, df['wasserstein'], color=color_w, s=10, label='Wasserstein')

    ax.plot(x, df['mae'], color=color_m)
    ax.scatter(x, df['mae'], color=color_m, s=10, label='MAE')

    ax.set_xlabel('Sampling temperature')
    ax.set_ylabel('Metric value')
    ax.set_ylim(0, 45)
    ax.set_yticks(range(0, 46, 15))
    ax.set_xlim(0, 2.1)
    ax.legend()

    ax.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('transformer.pdf', bbox_inches='tight')
    plt.show()
