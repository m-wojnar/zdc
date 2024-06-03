import matplotlib.pyplot as plt
import numpy as np

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    color_r = (1.0, 0.4, 0.4)
    color_g = (0.3, 0.3, 1.0)

    original = np.load('original.npz')['arr_0']
    generated = np.load('generated.npz')['arr_0']

    _, axs = plt.subplots(5, 1, figsize=(COLUMN_WIDTH, 1.5 * COLUMN_WIDTH))

    for i, ax in enumerate(axs):
        ax.hist(original[:, i], bins=200, color=color_r, label='Original', fill=True, alpha=0.75)
        ax.hist(generated[:, i], bins=200, color=color_g, label='Generated', fill=True, alpha=0.7)

        if i == 0:
            ax.legend()
        if i == 4:
            ax.set_xlabel('Photon count')
        else:
            ax.set_xticklabels([])

        ax.set_ylabel(f'Channel {i + 1}')
        ax.set_ylim((1, 1e5))
        ax.set_yscale('log')
        ax.set_xlim(-50, 1650)
        ax.grid(True)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('histogram.pdf', bbox_inches='tight')
    plt.show()
