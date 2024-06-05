import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HIGHT


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv('kl_loss.csv')
    latent_size = df.columns.astype(int).tolist()
    epochs = range(1, len(df) + 1)

    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0.1, 0.9, len(latent_size)))
    _, ax = plt.subplots(figsize=(0.8 * COLUMN_WIDTH, 0.8 * COLUMN_HIGHT))

    for size, color in zip(sorted(latent_size), colors):
        ax.plot(epochs, df[str(size)], color=color, label=size)

    plt.xlabel('Epoch')
    plt.ylabel(r'$D_{KL}$')
    plt.yticks(range(0, 21, 5))
    plt.xlim(0, 100)
    plt.ylim(0, 20)

    plt.legend(title='Latent space size', loc='upper right', ncol=2)
    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('kl_loss.pdf', bbox_inches='tight')
    plt.show()
