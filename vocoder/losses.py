import torch
from torchaudio.transforms import Spectrogram
import torch.nn as nn
import torch.nn.functional as F


class SpectralLoss(nn.Module):
    def __init__(self, nfft, alpha=1.0, overlap=0.75):
        super().__init__()
        self.nfft = nfft
        self.alpha = alpha
        self.overlap = overlap
        self.hopsize = int(nfft * (1 - overlap))
        self.spec = Spectrogram(n_fft=nfft, win_length=nfft, hop_length=self.hopsize, power=1)  # power=1, magnitude

    def forward(self, x_hat, x):
        """
        #### input ####
        x_hat   : the synthesized waveform, in shape [B, t*framesize]
        x       : the ground truth segments, in shape [B, t*framesize]

        #### output ####
        loss    : the spectral loss between `x_hat` and `x`

        """
        x_hat_spec = self.spec(x_hat)
        x_spec = self.spec(x)
        loss = F.l1_loss(x_hat_spec, x_spec) + self.alpha * F.l1_loss(
            torch.log(x_hat_spec + 1e-7), torch.log(x_spec + 1e-7)
        )
        return loss


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self, nffts=[2048, 1024, 512, 256, 128, 64], alpha=1.0, overlap=0.75):
        super().__init__()
        self.losses = nn.ModuleList([SpectralLoss(nfft, alpha, overlap) for nfft in nffts])

    def forward(self, x_hat, x):
        out = 0.0
        for loss in self.losses:
            out += loss(x_hat, x)
        return out

