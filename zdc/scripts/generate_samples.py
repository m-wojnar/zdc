import jax
import matplotlib.pyplot as plt

from zdc.architectures.vit import Encoder, Decoder
from zdc.models.autoencoder.variational import VAE
from zdc.utils.data import get_samples
from zdc.utils.nn import load_model
from zdc.utils.train import default_generate_fn


if __name__ == '__main__':
    model = VAE(Encoder, Decoder)
    params, state = load_model('../models/autoencoder/checkpoints/variational_vit/epoch_100.pkl.lz4')

    test_key = jax.random.PRNGKey(42)
    responses, particles = get_samples()
    generated = default_generate_fn(model)(params, state, test_key, responses, particles)

    n = 7
    fig, axs = plt.subplots(2, n, figsize=(2 * n + 1, 4), dpi=200)

    for i in range(2 * n):
        x = responses[i] if i < n else generated[i - n]
        ax = axs[i // n, i % n]
        im = ax.imshow(x, interpolation='none', cmap='gnuplot')
        ax.axis('off')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('generated.png', bbox_inches='tight')
    plt.show()
