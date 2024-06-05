import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from zdc.scripts.plots import PLOT_PARAMS, COLUMN_WIDTH, COLUMN_HIGHT


if __name__ == '__main__':
    plt.rcParams.update(PLOT_PARAMS)
    plt.rcParams['legend.fontsize'] = 6

    df = pd.read_csv('perplexity_update.csv')
    epochs = range(1, len(df) + 1)

    update_rules = df.columns.tolist()
    rule_names = {
        'ema': r'EMA',
        'ema_l2': r'EMA + $l2$',
        'ema_l2_proj': r'EMA + $l2$ + proj',
        'ema_proj': r'EMA + proj',
        'grad': r'Grad',
        'grad_l2': r'Grad + $l2$',
        'grad_l2_proj': r'Grad + $l2$ + proj',
        'grad_proj': r'Grad + proj',
    }

    order = np.argsort(df.values.max(axis=0))
    update_rules = [update_rules[i] for i in order]

    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0.1, 0.9, len(rule_names)))
    _, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_HIGHT))

    for rule, color in zip(update_rules, colors):
        ax.plot(epochs, df[rule], color=color, label=rule_names[rule])

    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.yticks(range(0, 151, 30))
    plt.xlim(0, 100)
    plt.ylim(0, 150)

    plt.legend(ncol=2, bbox_to_anchor=(0.33, 0.38))
    plt.grid(True)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('perplexity_update.pdf', bbox_inches='tight')
    plt.show()
