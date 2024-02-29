from typing import Callable

from flax import linen as nn


class MLP(nn.Module):
    layer_sizes: list
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.Dense(size)(x)
            x = self.activation(x)

        return nn.Dense(self.layer_sizes[-1])(x)
