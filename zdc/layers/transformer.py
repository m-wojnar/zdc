from flax import linen as nn


class FeedForwardBlock(nn.Module):
    hidden_dim: int
    drop_rate: float

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim)(x)
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=not training)
        return x


class TransformerEncoderBlock(nn.Module):
    num_heads: int
    hidden_dim: int
    drop_rate: float
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x, training=True):
        residual = x
        x = nn.LayerNorm(self.epsilon)(x)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=x.shape[-1])(x)
        x = x + residual

        residual = x
        x = nn.LayerNorm(self.epsilon)(x)
        x = FeedForwardBlock(self.hidden_dim, self.drop_rate)(x, training=training)
        x = x + residual

        return x
