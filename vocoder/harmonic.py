import torch
import torch.nn as nn
import torch.nn.functional as F


class HarmonicOscillator(nn.Module):
    def __init__(self, fs=16000, framesize=80):
        """
        fs          : the sampling rate of the output audio, default to 16kHz
        framesize   : the number of points contained in one frame, default to 80, i.e. f_model = sr / framesize = 200Hz

        """
        super().__init__()
        self.fs = fs
        self.framesize = framesize

    def upsample(self, s):
        """
        #### input ####
        s   : in shape [B, C, t], sampled at f_model

        #### output ####
        out : in shape [B, C, t*framesize], upsampled to fs, i.e. upsample by a factor of framesize
        """
        B, C, t = s.shape
        # define upsample window
        winsize = 2 * self.framesize + 1
        window = torch.hann_window(winsize, device=s.device).float()
        kernel = window.expand(C, 1, winsize)  # [C, 1, winsize]
        # prepare zero inserted buffer
        buffer = torch.zeros(B, C, t * self.framesize, device=s.device).float()
        buffer[:, :, :: self.framesize] = s  # [B, C, t']
        # use 1d depthwise conv to upsample
        out = F.conv1d(buffer, kernel, padding=self.framesize, groups=C)  # [B, C, t']
        return out

    def forward(self, f0, cn, an):
        """
        #### input ####
        f0  : in shape [B, 1, t], calculated at f_model
        cn  : in shape [B, nharmonics, t], calculated at f_model
        an  : in shape [B, 1, t], calculated at f_model

        #### output ####
        out : in shape [B, t*framesize], calculated at fs

        """
        # build harmonics
        f0 = F.interpolate(f0, scale_factor=self.framesize, mode="linear")  # [B, 1, t*framesize]
        nharmonics = cn.shape[1]
        harmonics_range = (
            torch.linspace(1, nharmonics, nharmonics, dtype=torch.float32).unsqueeze(-1).to(cn.device)
        )  # [nharmonics, 1]
        harmonics = harmonics_range @ f0  # [B, nharmonics, t*framesize]

        # upsample cn, an using hann window
        cn = self.upsample(cn)  # [B, nharmonics, t*framesize]
        an = self.upsample(an)  # [B, 1, t*framesize]

        # anti-alias masking
        mask = (harmonics < self.fs / 2).float()
        cn = cn.masked_fill(mask == 0, float("-1e20"))
        cn = F.softmax(cn, dim=1)  # [B, nharmonics, t*framesize]

        # build phase
        phi = 2 * torch.pi * torch.cumsum(harmonics / self.fs, dim=-1)  # [B, nharmonics, t*framesize]

        # build output
        out = (an * cn * torch.sin(phi)).sum(1)  # [B, t*framesize]
        return out


if __name__ == "__main__":
    oscillator = HarmonicOscillator()
    B = 4
    t = 100
    nharmonics = 60
    f0 = torch.randn(B, t).to("cuda")
    cn = torch.randn(B, nharmonics, t).to("cuda")
    an = torch.randn(B, t).to("cuda")

    cn.requires_grad_(True)
    an.requires_grad_(True)

    out = oscillator(f0, cn, an)
    print(out.shape)
    print(out.requires_grad)
