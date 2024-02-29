import jax
from flax import linen as nn


class UpSample(nn.Module):
    scale: int = 2
    method: str = 'nearest'

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        return jax.image.resize(x, (b, h * self.scale, w * self.scale, c), method=self.method)
