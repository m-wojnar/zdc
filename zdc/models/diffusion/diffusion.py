import os

import jax
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMPipeline, DDIMScheduler, ImagePipelineOutput, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils.torch_utils import randn_tensor
from tqdm import trange

from zdc.models import RESPONSE_SHAPE, PARTICLE_SHAPE
from zdc.utils.data import load, get_samples
from zdc.utils.metrics import Metrics
from zdc.utils.torch import get_data_loader, torch_to_numpy
from zdc.utils.train import default_eval_fn


class DDIMConditionPipeline(DDIMPipeline):
    @torch.no_grad()
    def __call__(self, cond, generator, num_inference_steps=50, eta=1.0, guidance_scale=1.0):
        batch_size = cond.shape[0]

        image_shape = (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size)
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        self.scheduler.set_timesteps(num_inference_steps)

        if guidance_scale > 1.0:
            cond = torch.cat([cond] * 2)
            neg_mask = torch.zeros((batch_size, 1), device=self._execution_device)
            pos_mask = torch.ones((batch_size, 1), device=self._execution_device)
            attention_mask = torch.cat([neg_mask, pos_mask])

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1.0:
                model_output = self.unet(torch.cat([image] * 2), t, cond, encoder_attention_mask=attention_mask).sample
                neg_output, pos_output = model_output.chunk(2)
                model_output = (guidance_scale + 1.0) * pos_output - guidance_scale * neg_output
            else:
                model_output = self.unet(image, t, cond).sample

            image = self.scheduler.step(
                model_output, t, image, eta=eta, generator=generator, use_clipped_model_output=True
            ).prev_sample

        image = torch.relu(image)
        return ImagePipelineOutput(images=torch_to_numpy(image))


def get_model(device):
    return UNet2DConditionModel(
        sample_size=RESPONSE_SHAPE[0],
        in_channels=RESPONSE_SHAPE[-1],
        out_channels=RESPONSE_SHAPE[-1],
        layers_per_block=2,
        block_out_channels=(16, 32, 64, 128),
        down_block_types=(
            'DownBlock2D',
            'DownBlock2D',
            'CrossAttnDownBlock2D',
            'DownBlock2D',
        ),
        mid_block_type='UNetMidBlock2DCrossAttn',
        up_block_types=(
            'UpBlock2D',
            'CrossAttnUpBlock2D',
            'UpBlock2D',
            'UpBlock2D',
        ),
        encoder_hid_dim=PARTICLE_SHAPE[0],
        cross_attention_dim=32,
        attention_head_dim=4,
        norm_num_groups=8
    ).to(device)


def get_noise_scheduler():
    return DDIMScheduler(num_train_timesteps=1000, clip_sample=True, clip_sample_range=7.0, timestep_spacing='trailing')


def get_optimizer(model, n_examples=214746, epochs=100, batch_size=256):
    train_steps = epochs * n_examples // batch_size

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.2 * train_steps),
        num_training_steps=train_steps
    )
    return optimizer, lr_scheduler


def train_loop(
        name, model, noise_scheduler, optimizer, lr_scheduler, train_dataloader, val_dataloader, test_dataloader,
        generator, device, epochs=100, n_rep_val=1, n_rep_test=5
):
    metrics = Metrics(job_type='train', name=name)
    os.makedirs(f'checkpoints/{name}', exist_ok=True)

    eval_fn = jax.jit(default_eval_fn)
    eval_metrics = ('mse', 'mae', 'wasserstein')

    samples = get_samples()
    samples = (samples[0], torch.Tensor(samples[1]).to(device)[:, None])

    for epoch in trange(epochs, desc='Epochs'):
        model.train()

        for clean_images, cond in train_dataloader:
            optimizer.zero_grad()

            noise = torch.randn(clean_images.shape, generator=generator).to(device)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.shape[0],), generator=generator).to(device).long()
            keep_cond = torch.rand((cond.shape[0], 1), generator=generator).to(device) > 0.15

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            noise_pred = model(noisy_images, timesteps, cond, encoder_attention_mask=keep_cond).sample

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()

            metrics.add({'mse_noise': loss.detach().item()}, 'train')

        metrics.log(epoch)
        generated, original = [], []

        model.eval()

        with torch.no_grad():
            pipeline = DDIMConditionPipeline(unet=model, scheduler=noise_scheduler)

            for clean_images, cond in val_dataloader:
                for _ in range(n_rep_val):
                    generated.append(pipeline(cond, generator).images)
                    original.append(clean_images)

            generated, original = np.concatenate(generated, dtype=np.float32), (np.concatenate([torch_to_numpy(xs) for xs in original], dtype=np.float32),)
            metrics.add(dict(zip(eval_metrics, eval_fn(generated, *original))), 'val')
            metrics.log(epoch)

            responses = pipeline(samples[1], generator=torch.manual_seed(0)).images
            metrics.plot_responses(samples[0], responses, epoch)
            metrics.log(epoch)

            pipeline.save_pretrained(f'checkpoints/{name}/epoch_{epoch}')

    generated, original = [], []

    with torch.no_grad():
        pipeline = DDIMConditionPipeline(unet=model, scheduler=noise_scheduler)

        for clean_images, cond in test_dataloader:
            for _ in range(n_rep_test):
                generated.append(pipeline(cond, generator).images)
                original.append(clean_images)

        generated, original = np.concatenate(generated, dtype=np.float32), (np.concatenate([torch_to_numpy(xs) for xs in original], dtype=np.float32),)
        metrics.add(dict(zip(eval_metrics, eval_fn(generated, *original))), 'test')
        metrics.log(epochs)


if __name__ == '__main__':
    train_batch_size, eval_batch_size = 256, 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100
    generator = torch.manual_seed(42)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    train_dataloader = get_data_loader(r_train, p_train, train_batch_size, generator, device, shuffle=True)
    val_dataloader = get_data_loader(r_val, p_val, eval_batch_size, generator, device, shuffle=True)
    test_dataloader = get_data_loader(r_test, p_test, eval_batch_size, generator, device, shuffle=False)

    model = get_model(device)
    noise_scheduler = get_noise_scheduler()
    optimizer, lr_scheduler = get_optimizer(model, epochs=epochs, batch_size=train_batch_size)

    print("Number of parameters:", model.num_parameters())

    train_loop(
        'diffusion', model, noise_scheduler, optimizer, lr_scheduler,
        train_dataloader, val_dataloader, test_dataloader, generator, device, epochs
    )
