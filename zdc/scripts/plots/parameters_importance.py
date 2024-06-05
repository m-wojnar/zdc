import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS


param_names = {
    'learning_rate': 'Learning rate',
    'beta_1': r'$\beta_1$',
    'beta_2': r'$\beta_2$',
    'epsilon': r'$\epsilon$',
    'cosine_decay': 'Cosine decay',
    'nesterov': 'Nesterov momentum',
    'use_weight_decay': 'Weight decay'
}

arch_names = {
    'cnn': 'CNN',
    'vit': 'ViT',
    'mlp': 'MLP-Mixer'
}


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)

    df = pd.read_csv('importance.csv')
    n = len(arch_names)
    parameters = df.columns[1:]
    parameters = [param_names.get(p, p) for p in parameters]

    y_vals = np.arange(len(parameters))[::-1]
    y_offset = np.linspace(0, 0.4, n)[::-1]

    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0.1, 0.9, n))
    _, ax = plt.subplots()

    for arch, offset, color in zip(arch_names, y_offset, colors):
        values = df[df['architecture'] == arch].values[0, 1:]
        ax.barh(y_vals + offset, values, 0.15, label=arch_names[arch], color=color)

    ax.set_yticks(y_vals + (y_offset[0] / 2), parameters)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlim(0, 1)
    ax.set_xlabel('Importance')

    ax.legend()
    ax.grid(axis='x')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('importance.pdf', bbox_inches='tight')
    plt.show()
