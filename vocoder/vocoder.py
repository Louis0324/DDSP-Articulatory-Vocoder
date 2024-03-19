import torch
import torch.nn as nn
import math
from encoder import Encoder
from harmonic import HarmonicOscillator
from noise import FilteredNoiseGenerator


class Vocoder(nn.Module):
    def __init__(
        self, hidden_dim, nharmonics, nbands, attenuate=0.02, fs=16000, framesize=80
    ):
        super().__init__()
        self.encoder = Encoder(hidden_dim, nharmonics, nbands, attenuate)
        self.harmonic = HarmonicOscillator(fs, framesize)
        self.noise = FilteredNoiseGenerator(framesize)

    def forward(self, f0, loudness, ema):
        """
        #### input ####
        f0          : in shape [B, 1, t], calculated at f_model
        loudness    : in shape [B, 1, t], calculated at f_model
        ema         : in shape [B, 12, t], sampled at f_model

        #### output ####
        speech      : in shape [B, t*framesize]

        """
        # going through encoder to get the control signals
        cn, an, H = self.encoder(f0, loudness, ema)
        # generate harmonic components
        harmonics = self.harmonic(f0, cn, an)  # [B, t*framesize]
        # generate filtered noise
        noise = self.noise(H)  # [B, t*framesize]
        # additive synthesis
        speech = harmonics + noise  # [B, t*framesize]
        return speech


if __name__ == "__main__":
    hidden_dim = 512
    nharmonics = 100
    nbands = 89
    vocoder = Vocoder(hidden_dim, nharmonics, nbands).to("cuda")

    B = 4
    t = 100
    f0 = torch.randn(B, 1, t).to("cuda")
    loudness = torch.randn(B, 1, t).to("cuda")
    ema = torch.randn(B, 12, t).to("cuda")

    speech = vocoder(f0, loudness, ema)
    print(speech.shape, speech.requires_grad)
