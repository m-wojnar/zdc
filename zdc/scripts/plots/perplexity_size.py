import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HIGHT


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv('perplexity_size.csv')
    epochs = range(1, len(df) + 1)

    sizes = df.columns.astype(int).tolist()

    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0.1, 0.9, len(sizes)))
    _, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_HIGHT))

    for size, color in zip(sorted(sizes), colors):
        ax.plot(epochs, df[str(size)], color=color, label=size)

    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.yticks(range(0, 201, 40))
    plt.xlim(0, 100)
    plt.ylim(0, 200)

    plt.legend(title='Codebook size', ncol=3)
    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('perplexity_size.pdf', bbox_inches='tight')
    plt.show()
