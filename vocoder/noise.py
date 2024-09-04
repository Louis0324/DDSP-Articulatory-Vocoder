import torch
import torch.nn as nn
from torchaudio.functional import fftconvolve

class FilteredNoiseGenerator(nn.Module):
    def __init__(self, framesize=80):
        super().__init__()
        self.framesize = framesize
        
    def forward(self, H):
        """Generate time-varying filtered noise by overlap-add method using linear phase LTV-FIR filter
        #### input ####
        H   : half of the frequency response of the zero-phase filter, all real numbers
                in shape [B, t, nbands], sampled at f_model = fs / framesize, default to fs = 16kHz, f_model = 200Hz
        
        #### output ####
        out : in shape [B, t*framesize]
    
        """
        
        B, t, nbands = H.shape
        
        # irfft to get zero-phase filter
        zero_phase = torch.fft.irfft(H, n=2*nbands-1, dim=-1) # [B, t, 2*nbands-1]
        
        # shift zero-phase filter to causal
        lin_phase = zero_phase.roll(nbands-1, -1) 
        
        # window the filter in the time domain
        hann = torch.hann_window(2*nbands - 1).float().to(H.device)
        firwin = lin_phase * hann # [B, t, 2*nbands-1]
        
        # generate noise
        noise = torch.rand(B, t, self.framesize).float().to(H.device) * 2 - 1 # [B, t, framesize]
        
        # fftconvolve
        filtered_noise = fftconvolve(noise, firwin) # [B, t, framesize+2*nbands-2]
        
        # Overlap-add to build time-varying filtered noise.
        eye = torch.eye(filtered_noise.shape[-1]).unsqueeze(1).to(H.device) # [framesize+2*nbands-2, 1, framesize+2*nbands-2]
        out = nn.functional.conv_transpose1d(filtered_noise.transpose(1, 2), eye, stride=self.framesize).squeeze(1) # [B, t*framesize+2*nbands-2]
        out = out[:, nbands-1:-(nbands-1)] # [B, t*framesize]
        return out
    

