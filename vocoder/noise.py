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
        
        regarding the choice of `nbands`, under default settings where framesize = 80, i.e. each noise segment is of length 80, we want the zero-padded segment to be of length 256, 
        i.e. the filter length should be 256-80+1 = 177 = 2*nbands-1, and before doing irfft, nbands should be (177+1)/2 = 89
        
        (could be too much computation here, framesize could be larger)
        
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
    
    
if __name__ == '__main__':
    Noise = FilteredNoiseGenerator()
    B = 2
    t = 500
    nbands = 89
    H = torch.randn(B, t, nbands).to('cuda')
    H.requires_grad_(True)
    out = Noise(H)
    print(out.shape, out.requires_grad)
