from flax import linen as nn


class DenseBlock(nn.Module):
    features: int
    use_bn: bool = False
    dropout_rate: float = None
    negative_slope: float = None

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Dense(self.features)(x)

        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not training)(x)
        if self.dropout_rate is not None:
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        if self.negative_slope is not None:
            x = nn.leaky_relu(x, negative_slope=self.negative_slope)

        return x
