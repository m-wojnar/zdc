from flax import linen as nn


class ConvBlock(nn.Module):
    features: int
    kernel_size: int = 3
    strides: int = 1
    padding: str = 'same'
    use_bn: bool = False
    dropout_rate: float = None
    negative_slope: float = None
    max_pool_size: int = None

    @nn.compact
    def __call__(self, x, training=True):
        x = nn.Conv(self.features, kernel_size=(self.kernel_size, self.kernel_size), strides=(self.strides, self.strides), padding=self.padding)(x)

        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not training)(x)
        if self.dropout_rate is not None:
            x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        if self.negative_slope is not None:
            x = nn.leaky_relu(x, negative_slope=self.negative_slope)
        if self.max_pool_size is not None:
            pool_size = (self.max_pool_size, self.max_pool_size)
            x = nn.max_pool(x, window_shape=pool_size, strides=pool_size)

        return x
