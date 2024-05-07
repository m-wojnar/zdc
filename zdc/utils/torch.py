import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data_loader(responses, particles, batch_size, generator, device, shuffle=True):
    responses = torch.Tensor(np.array(responses)).to(device)
    particles = torch.Tensor(np.array(particles)).to(device)

    responses = responses.permute(0, 3, 1, 2)
    particles = particles[:, None]

    dataset = TensorDataset(responses, particles)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def torch_to_numpy(x):
    return x.cpu().permute(0, 2, 3, 1).numpy()
