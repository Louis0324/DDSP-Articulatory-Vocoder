import torch
from torch import nn
import math
from utils import DilatedConvEncoder      

class EncoderDilatedConvSinCos(nn.Module):
    def __init__(self, hidden_dim=256, nharmonics=50, nbands=65, attenuate=0.01, dilations=[1, 2, 4, 8, 16], nstacks=4):
        super().__init__()        
        self.attenuate = attenuate 
        self.nharmonics = nharmonics
        self.hidden_dim = hidden_dim

        self.convencoder = DilatedConvEncoder(in_channels=14, out_channels=hidden_dim, kernel_size=3, stride=1, dilations=dilations, nstacks=nstacks)
        self.conditioner = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, padding=1)
        )
        
        self.head_amp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (nharmonics+1)*2)
        )
        self.head_H = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nbands)
        )
        

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
        cn_sin      : in shape [B, nharmonics, t], calculated at f_model
        cn_cos      : in shape [B, nharmonics, t], calculated at f_model
        an_sin      : in shape [B, 1, t], calculated at f_model 
        an_cos      : in shape [B, 1, t], calculated at f_model 
        H           : in shape [B, t, nbands], calculated at f_model
        
        """
        # normalize f0 to be within [0, 1]
        f0 = f0 / 500
        
        # dilated conv encoding
        in_feat = torch.concat([f0, loudness, ema], dim=1) # [B, 14, t]
        out_feat = self.convencoder(in_feat).transpose(1,2) # [B, t, hidden_dim]

        # loudness FiLM
        condition = self.conditioner(loudness).transpose(1,2) # [B, t, hidden_dim*2]
        out_feat = out_feat * condition[:,:,:self.hidden_dim] + condition[:,:,self.hidden_dim:]
            
        # output heads
        amp = self.head_amp(out_feat) # [B, t, (nharmonics+1)*2]
        amp_sin = amp[:,:,:(self.nharmonics+1)] # [B, t, nharmonics+1]
        amp_cos = amp[:,:,(self.nharmonics+1):] # [B, t, nharmonics+1]
        
        cn_sin = (amp_sin[:, :, 1:]).transpose(1,2) # [B, nharmonics, t]
        an_sin = self._scaled_sigmoid(amp_sin[:, :, 0].unsqueeze(-1)).transpose(1,2) # [B, 1, t]
        
        cn_cos = (amp_cos[:, :, 1:]).transpose(1,2) # [B, nharmonics, t]
        an_cos = self._scaled_sigmoid(amp_cos[:, :, 0].unsqueeze(-1)).transpose(1,2) # [B, 1, t]
        
        H  = self._scaled_sigmoid(self.head_H(out_feat))*self.attenuate

        return cn_sin, cn_cos, an_sin, an_cos, H

