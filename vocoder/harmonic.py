import torch
import torch.nn as nn
import torch.nn.functional as F

class HarmonicOscillatorSinCos(nn.Module):
    def __init__(self, fs=16000, framesize=80):
        """
        fs          : the sampling rate of the output audio, default to 16kHz
        framesize   : the number of points contained in one frame, default to 80, i.e. f_model = sr / framesize = 200Hz
        
        """
        super().__init__()
        self.fs = fs
        self.framesize = framesize
        self.winsize = 2 * self.framesize + 1
        self.hann = torch.hann_window(self.winsize).float()

    def upsample(self, s):
        """
        #### input ####
        s   : in shape [B, C, t], sampled at f_model
        
        #### output ####
        out : in shape [B, C, t*framesize], upsampled to fs, i.e. upsample by a factor of framesize
        """
        B, C, t = s.shape
        # define upsample window
        winsize = 2*self.framesize+1
        window = torch.hann_window(winsize, device=s.device).float()
        kernel = window.expand(C, 1, winsize) # [C, 1, winsize]
        # prepare zero inserted buffer
        buffer = torch.zeros(B, C, t*self.framesize, device=s.device).float()
        buffer[:,:,::self.framesize] = s # [B, C, t']
        # use 1d depthwise conv to upsample
        out = F.conv1d(buffer, kernel, padding=self.framesize, groups=C) # [B, C, t']
        return out
    
    def upsample_cpu(self, s):
        """
        #### input ####
        s   : in shape [B, C, t], sampled at f_model
        
        #### output ####
        out : in shape [B, C, t*framesize], upsampled to fs, i.e. upsample by a factor of framesize
        """
        B, C, t = s.shape
        
        # define upsample window
        kernel = self.hann.to(s.device).expand(B, C, self.winsize) # [B, C, winsize]
        
        # prepare buffer
        buffer = torch.zeros(B, C, (t-1)*self.framesize+self.winsize, device=s.device).float()
        
        # use for loop to fill in the upsampled values
        for i in range(t):
            start = i * self.framesize
            end = start + self.winsize 
            buffer[:, :, start:end] += s[:, :, i:i+1] * kernel
        
        out = buffer[:,:,self.framesize+1:]
        return out
        
    def forward(self, f0, cn_sin, cn_cos, an_sin, an_cos):
        """
        #### input ####
        f0  : in shape [B, 1, t], calculated at f_model 
        cn  : in shape [B, nharmonics, t], calculated at f_model
        an  : in shape [B, 1, t], calculated at f_model 

        #### output ####
        out : in shape [B, t*framesize], calculated at fs

        """
        device = f0.device
        
        # build harmonics
        f0 = F.interpolate(f0, scale_factor=80, mode='linear') # [B, 1, t*framesize]
        nharmonics = cn_sin.shape[1]
        harmonics_range = torch.linspace(1, nharmonics, nharmonics, dtype=torch.float32).unsqueeze(-1).to(cn_sin.device) # [nharmonics, 1]
        harmonics = harmonics_range @ f0 # [B, nharmonics, t*framesize]
        
        # upsample cn, an using hann window 
        if device == torch.device('cpu'):
            cn_sin = self.upsample_cpu(cn_sin) # [B, nharmonics, t*framesize]
            cn_cos = self.upsample_cpu(cn_cos)
            an_sin = self.upsample_cpu(an_sin) # [B, 1, t*framesize]
            an_cos = self.upsample_cpu(an_cos)
        else:
            cn_sin = self.upsample(cn_sin) # [B, nharmonics, t*framesize]
            cn_cos = self.upsample(cn_cos)
            an_sin = self.upsample(an_sin) # [B, 1, t*framesize]
            an_cos = self.upsample(an_cos)
        
        # anti-alias masking 
        mask = (harmonics < self.fs / 2).float()
        cn_sin = cn_sin.masked_fill(mask == 0, float("-1e20"))
        cn_cos = cn_cos.masked_fill(mask == 0, float("-1e20"))
        cn_sin = F.softmax(cn_sin, dim=1) # [B, nharmonics, t*framesize]
        cn_cos = F.softmax(cn_cos, dim=1) # [B, nharmonics, t*framesize]
        
        # build phase
        phi = 2*torch.pi*torch.cumsum(harmonics / self.fs, dim=-1) # [B, nharmonics, t*framesize]
        
        # build output
        out = (an_sin * cn_sin * torch.sin(phi)).sum(1) + (an_cos * cn_cos * torch.cos(phi)).sum(1) # [B, t*framesize]
        return out