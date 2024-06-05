import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HIGHT


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv('model_codebook_size.csv')
    cols = map(lambda x: list(filter(lambda y: x in y, df.columns)), ['s_', 'xl_', 'm_rec_'])
    cols = list(map(lambda x: sorted(x, key=lambda y: int(y.split('_')[-1])), cols))

    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0.1, 0.9, 3))
    _, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_HIGHT))

    for col, color, label in zip(cols, colors, ['0.25M - gen', '52M - gen', '1M - rec']):
        sizes = list(map(lambda x: int(x.split('_')[-1]), col))
        wasserstein = df[col].mean(axis=0)

        ax.plot(sizes, wasserstein, color=color, zorder=0)
        ax.scatter(None, None, color=color, label=label, zorder=0, s=15)

        for c, s, was in zip(col, sizes, wasserstein):
            ax.scatter(s, was, color=color, zorder=1, s=15)

    plt.xlabel('Codebook size')
    plt.ylabel('Wasserstein')
    plt.yticks(range(0, 16, 3))
    plt.xscale('log')
    plt.ylim(0, 15)

    plt.legend(title='Model size - task', loc='lower center')
    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('model_codebook_size.pdf', bbox_inches='tight')
    plt.show()
