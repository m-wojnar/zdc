import jax.numpy as jnp
from flax import linen as nn


class Patches(nn.Module):
    patch_size: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = x.reshape(b, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, -1, *x.shape[3:])
        x = x.reshape(b, x.shape[1], -1)
        return x


class Unpatch(nn.Module):
    patch_size: int
    h: int
    w: int

    @nn.compact
    def __call__(self, x):
        b, _, c = x.shape
        new_c = c // (self.patch_size ** 2)
        x = x.reshape(b, -1, self.patch_size, self.patch_size, new_c)
        x = x.reshape(b, self.h // self.patch_size, self.w // self.patch_size, self.patch_size, self.patch_size, new_c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, self.h, self.w, new_c)
        return x


class PatchEncoder(nn.Module):
    num_patches: int
    embedding_dim: int
    positional_encoding: bool = False

    @nn.compact
    def __call__(self, x, training=True):
        if self.positional_encoding:
            pos_embedding = nn.Embed(self.num_patches, self.embedding_dim)(jnp.arange(self.num_patches))
            x = nn.Dense(self.embedding_dim)(x)
            x = x + pos_embedding
        else:
            x = nn.Dense(self.embedding_dim)(x)

        return x


class PatchExpand(nn.Module):
    h: int
    w: int

    @nn.compact
    def __call__(self, x):
        b, _, c = x.shape
        x = x.reshape(b, self.h, self.w, c)
        x = nn.Dense(2 * c)(x)
        x = x.reshape(b, 2 * self.h, 2 * self.w, c // 2)
        x = nn.LayerNorm()(x)
        x = x.reshape(b, -1, c // 2)
        return x


class PatchMerge(nn.Module):
    h: int
    w: int

    @nn.compact
    def __call__(self, x):
        b, _, c = x.shape
        x = x.reshape(b, self.h, self.w, c)

        if self.h % 2 != 0 or self.w % 2 != 0:
            x = jnp.pad(x, ((0, 0), (0, self.h % 2), (0, self.w % 2), (0, 0)))

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)
        x = nn.LayerNorm()(x)
        x = nn.Dense(2 * c)(x)
        x = x.reshape(b, -1, 2 * c)
        return x
