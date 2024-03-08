from typing import Callable

from flax import linen as nn


class MLP(nn.Module):
    layer_sizes: list
    activation_fn: Callable = nn.relu
    activation_final_fn: Callable = None

    @nn.compact
    def __call__(self, x):
        for size in self.layer_sizes[:-1]:
            x = nn.Dense(size)(x)
            x = self.activation_fn(x)

        if self.activation_final_fn is not None:
            x = self.activation_final_fn(x)

        return nn.Dense(self.layer_sizes[-1])(x)


class MixerBlock(nn.Module):
    tokens_dim: int
    channels_dim: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm()(x)
        x = x.transpose(0, 2, 1)
        x = MLP([self.tokens_dim, x.shape[-1]], activation_fn=nn.gelu)(x)
        x = x.transpose(0, 2, 1)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = MLP([self.channels_dim, x.shape[-1]], activation_fn=nn.gelu)(x)
        x = x + residual

        return x
