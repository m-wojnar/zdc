from collections import defaultdict
from pprint import pprint

import jax.numpy as jnp
import matplotlib.pyplot as plt
import wandb


class Metrics:
    def __init__(self):
        self.metrics = defaultdict(list)

    def reset(self):
        self.metrics = defaultdict(list)

    def add(self, metrics):
        for name, value in metrics.items():
            self.metrics[name].append(value)

    def collect(self):
        self.metrics = {metric: jnp.mean(jnp.array(values)).item() for metric, values in self.metrics.items()}

    def log(self, step):
        wandb.log(self.metrics, step=step)
        print(f"Step: {step}")
        pprint(self.metrics)

    @staticmethod
    def plot_responses(responses, generated, n=7):
        fig, axs = plt.subplots(2, n, figsize=(2 * n + 1, 4), dpi=200)

        for i in range(2 * n):
            x = responses[i] if i < n else generated[i - n]
            ax = axs[i // n, i % n]
            im = ax.imshow(x, interpolation='none', cmap='gnuplot')
            ax.axis('off')
            fig.colorbar(im, ax=ax)

        wandb.log({'generated': wandb.Image(fig)})
        plt.close(fig)
