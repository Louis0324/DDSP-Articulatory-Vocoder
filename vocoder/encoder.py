import torch
from torch import nn
import torch.nn.functional as F
import math
from modules import *


class Encoder(nn.Module):
    def __init__(self, hidden_dim=256, nharmonics=50, nbands=65, attenuate=0.02):
        super().__init__()
        self.attenuate = attenuate
        self.conv_input = ConvNet(in_channels=14, d_model=hidden_dim * 2, kernel_size=3, num_blocks=2)
        self.lstm = nn.LSTM(
            hidden_dim * 2,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.head_amp = nn.Linear(hidden_dim * 2, nharmonics + 1)
        self.head_H = nn.Sequential(nn.Linear(hidden_dim * 2, nbands), nn.LayerNorm(nbands))

    # Scale sigmoid as per original DDSP paper
    def _scaled_sigmoid(self, x):
        return 2.0 * (torch.sigmoid(x) ** math.log(10)) + 1e-7

    def forward(self, f0, loudness, ema):
        """
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model
        loudness    : in shape [B, 1, t], calculated at f_model
        ema         : in shape [B, 12, t], sampled at f_model

        #### output ####
        cn          : in shape [B, nharmonics, t], calculated at f_model
        an          : in shape [B, 1, t], calculated at f_model
        H           : in shape [B, t, nbands], calculated at f_model

        """
        # normalize f0 to be zero mean and unit variance
        f0 = (f0 - torch.mean(f0, dim=-1, keepdim=True)) / (torch.std(f0, dim=-1, keepdim=True) + 1e-7)

        # reshape
        f0 = f0.transpose(1, 2)  # [B, t, 1]
        loudness = loudness.transpose(1, 2)  # [B, t, 1]
        ema = ema.transpose(1, 2)  # [B, t, 12]

        # conv mapping
        in_feat = torch.concat([f0, loudness, ema], dim=-1)  # [B, t, 14]
        lstm_input = self.conv_input(in_feat.transpose(1, 2)).transpose(1, 2)  # [B, t, hidden_dim*2]

        # going through lstm
        lstm_output, _ = self.lstm(lstm_input)  # [B, t, hidden_dim*2]

        # output heads
        amp = self.head_amp(lstm_output)  # [B, t, nharmonics+1]
        cn = (amp[:, :, 1:]).transpose(1, 2)  # [B, nharmonics, t]
        an = self._scaled_sigmoid(amp[:, :, 0].unsqueeze(-1)).transpose(1, 2)
        H = self._scaled_sigmoid(self.head_H(lstm_output)) * self.attenuate

        return cn, an, H


if __name__ == "__main__":
    encoder = Encoder()
    nparam = 0
    for p in encoder.parameters():
        if p.requires_grad:
            nparam += p.numel()
    print(f"number of parameters: {nparam}")

    B = 4
    t = 200
    f0 = torch.randn(B, 1, t)
    loudness = torch.randn(B, 1, t)
    ema = torch.randn(B, 12, t)

    cn, an, H = encoder(f0, loudness, ema)
    print(cn.shape, an.shape, H.shape)
