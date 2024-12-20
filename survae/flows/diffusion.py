import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Iterable
from survae.distributions import Distribution
from survae.transforms import Transform

from tqdm import tqdm

class Flow(Distribution):
    """
    Base class for Flow.
    Flows use the forward transforms to transform data to noise.
    The inverse transforms can subsequently be used for sampling.
    These are typically useful as generative models of data.
    """

    def __init__(self, base_dist, transforms):
        super(Flow, self).__init__()
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.lower_bound = any(transform.lower_bound for transform in transforms)

    def log_prob(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)
        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj
        log_prob += self.base_dist.log_prob(x)
        return log_prob

    def sample_with_log_prob(self, num_samples):
        raise RuntimeError("Flow does not support sample_with_log_prob, see InverseFlow instead.")
    
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + self.shortcut(identity)
        out = self.relu(out)
        
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim: int, max_seq_len: int):
        super().__init__()

        pe = torch.zeros(max_seq_len, emb_dim)
        positions = torch.arange(0, max_seq_len).unsqueeze(1)
        div = torch.exp(
            -torch.log(torch.tensor(10000)) * torch.arange(0, emb_dim, 2) / emb_dim
        )  # sin (pos / 10000 ** (2i / emb_dim))
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :]

    def sample(self, x):
        return self.pe[x, :]
    
class ResNetWithTimeEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, max_seq_len=100):
        super().__init__()
        self.resnet1 = ResNetBlock(in_channels, out_channels)
        self.resnet2 = ResNetBlock(out_channels, out_channels)
        self.pe = PositionalEncoding(emb_dim=out_channels, max_seq_len=max_seq_len)
        
    def forward(self, x: torch.FloatTensor, t: torch.LongTensor) -> torch.FloatTensor:
        x = self.resnet2(self.resnet1(x))
        time_embed = self.pe.sample(t)
        emb = time_embed[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        
        return x + emb
    
class Down(nn.Module):
    def __init__(self, layers, max_seq_len=100):
        super().__init__()
        self.conv_layers = nn.ModuleList([ResNetWithTimeEmbed(dim_in, dim_out, max_seq_len=max_seq_len) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        self.bns = nn.ModuleList([nn.BatchNorm2d(feature_len) for feature_len in layers[1:-1]])
        self.down = nn.MaxPool2d(2)
        
    def forward(self, x, t):
        for layer, batch_norm in zip(self.conv_layers[:-1], self.bns):
            x = self.down(batch_norm(F.gelu(layer(x, t))))
        return self.down(self.conv_layers[-1](x, t))
    
class Up(nn.Module):
    def __init__(self, layers, max_seq_len=100):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_layers = nn.ModuleList([ResNetWithTimeEmbed(dim_in, dim_out, max_seq_len=max_seq_len) for dim_in, dim_out in zip(layers[:-1], layers[1:])])
        self.bns = nn.ModuleList([nn.BatchNorm2d(feature_len) for feature_len in layers[1:-1]])
        
    def forward(self, x, t):
        for layer, batch_norm in zip(self.conv_layers[:-1], self.bns):
            x = F.gelu(batch_norm(layer(self.up(x), t)))
        return self.conv_layers[-1](self.up(x), t)
    
class UNet(nn.Module):
    def __init__(self, layers, max_seq_len=100):
        super().__init__()
        self.up = Up(layers[::-1], max_seq_len=max_seq_len)
        self.down = Down(layers, max_seq_len=max_seq_len)
        
        self.down_outputs = []
        
        self.up.up.register_forward_hook(lambda module, input, output: output + self.down_outputs.pop(-1))
            
        for module in self.down.conv_layers.children():
            module.register_forward_hook(lambda module, input, output: self.down_outputs.append(output))
        
    def forward(self, x, t):
        return self.up(self.down(x, t), t)

class Diffusion(Distribution):
    def __init__(self, base_dist, model, transforms=[], transforms_schedule=[], num_timesteps=100, beta_start=0.0001, beta_end=0.02):
        super().__init__()

        # Flow initialization
        assert isinstance(base_dist, Distribution)
        if isinstance(transforms, Transform):
            transforms = [transforms]

        assert isinstance(transforms, Iterable)
        assert isinstance(transforms_schedule, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)
        self.transforms_schedule = transforms_schedule
        self.lower_bound = any(transform.lower_bound for transform in transforms)

        # Diffusion initialization
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1,0), value=1.)

        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.model = model
        
    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        return s1 * x_t - s2 * noise
    
    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x0, noise, t):
        s1 = self.sqrt_alphas_cumprod[t]
        s2 = self.sqrt_one_minus_alphas_cumprod[t]

        s1 = s1.reshape(-1, *[1,]*len(x0.shape[1:]))
        s2 = s1.reshape(-1, *[1,]*len(x0.shape[1:]))

        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def sample(self, num_samples):
        self.model.eval()
        z = self.base_dist.sample(num_samples)
        timesteps = list(range(self.num_timesteps))[::-1]
        for t in tqdm(timesteps):
            t = torch.from_numpy(np.repeat(t,  num_samples)).long()
            residual = self.model(z, t)
            z = self.step(residual, t[0], z)
        return z
    
    def loss(self, batch):
        timesteps = torch.randint(
            0, self.num_timesteps, (batch.shape[0],)
        ).long()

        noise = torch.randn(batch.shape)
        noisy = self.add_noise(batch, noise, timesteps)
        noise_pred = self.model(noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)

        return loss
