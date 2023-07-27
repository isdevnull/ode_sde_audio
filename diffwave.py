# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)



def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding





class DiffusionEmbedding(nn.Module):
  def __init__(self):
    super().__init__()
    #self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    # if diffusion_step.dtype in [torch.int32, torch.int64]:
    #   x = self.embedding[diffusion_step]
    # else:
    #   x = self._lerp_embedding(diffusion_step)
    embedding = timestep_embedding(diffusion_step, 128)
    x = self.projection1(embedding)
    x = silu(x)
    x = self.projection2(x)
    # x = self.projection1(x)
    # x = silu(x)
    # x = self.projection2(x)
    # x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class ResidualBlock(nn.Module):
  def __init__(self, n_mels, residual_channels, dilation, uncond=False):
    '''
    :param n_mels: inplanes of conv1x1 for spectrogram conditional
    :param residual_channels: audio conv
    :param dilation: audio conv dilation
    :param uncond: disable spectrogram conditional
    '''
    super().__init__()
    self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    self.diffusion_projection = Linear(512, residual_channels)
    if not uncond: # conditional model
      self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
    else: # unconditional model
      self.conditioner_projection = None

    self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

  def forward(self, x, diffusion_step, conditioner=None):
    assert (conditioner is None and self.conditioner_projection is None) or \
           (conditioner is not None and self.conditioner_projection is not None)

    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
    y = x + diffusion_step
    if self.conditioner_projection is None: # using a unconditional model
      y = self.dilated_conv(y)
    else:
      conditioner = self.conditioner_projection(conditioner)
      y = self.dilated_conv(y) + conditioner

    gate, filter = torch.chunk(y, 2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=1)
    return (x + residual) / sqrt(2.0), skip


class DiffWave(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    self.input_projection = Conv1d(1, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding()
    if self.params.unconditional: # use unconditional model
      self.spectrogram_upsampler = None
    else:
      self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

    self.residual_layers = nn.ModuleList([
        ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, 1, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, spectrogram=None):
    assert (spectrogram is None and self.spectrogram_upsampler is None) or \
           (spectrogram is not None and self.spectrogram_upsampler is not None)
    x = audio
    x = self.input_projection(x)
    x = F.relu(x)

    if diffusion_step.dim() == 0:
      diffusion_step = diffusion_step.view(1)
    diffusion_step = self.diffusion_embedding(diffusion_step)
    if self.spectrogram_upsampler: # use conditional model
      spectrogram = self.spectrogram_upsampler(spectrogram)

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, spectrogram)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from dataset_utils import mel_spectrogram
    import torchaudio.transforms as T

    params = OmegaConf.create(
        dict(
            residual_channels=512,
            residual_layers=27,
            n_mels=80,
            unconditional=False,
            dilation_cycle_length=12,
        )
    )
    model = DiffWave(params) #.to("cuda:0")
    print(sum(p.numel() for p in model.parameters()))
    
    mel1 = mel_spectrogram(torch.randn(1, 32768).view(-1).numpy(), n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=256, win_size=1024, fmin=0, fmax=8000)
    print(mel1.shape)
    x = torch.randn(2, 1, 32768)
    
    x_to_mel = torch.nn.functional.pad(x, (int((1024 - 256) / 2), int((1024 - 256) / 2)), mode="reflect")
    print(x.shape)

    transform = T.MelSpectrogram(16000, n_fft=1024, win_length=1024, hop_length=256, f_min=0, f_max=8000, n_mels=80, center=False)
    mel = transform(x_to_mel.squeeze())
    print(mel.shape)
    y = model(x, torch.rand(2), mel)
    print(y.shape)