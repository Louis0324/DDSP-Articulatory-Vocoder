import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_leaky_relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        ]
        if use_leaky_relu:
            layers.append(nn.LeakyReLU(0.2))
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvBlock(in_channels=in_channels, out_channels=features[0], stride=2, use_leaky_relu=True))

        in_channels = features[0]
        for feature in features[1:]:
            self.layers.append(
                ConvBlock(in_channels, feature, stride=2, use_leaky_relu=True)
            )
            in_channels = feature
        
        self.layers.append(ConvBlock(features[-1], 1, stride=2, use_leaky_relu=False))

        self.initialize_param()
        self.apply_weight_norm()
        
    def initialize_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    def apply_weight_norm(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m = nn.utils.weight_norm(m)
    
    def forward(self, x, save_feat=False):
        feat = x
        if save_feat:
            features = []
            for layer in self.layers:
                feat = layer(feat)
                features.append(feat)
            return features
        else:
            for layer in self.layers:
                feat = layer(feat)
            return feat
    
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, nscales=3, in_channels=1, features=[32, 64, 128]):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for _ in range(nscales):
            self.discriminators += [Discriminator(in_channels, features)]
            
    def forward(self, xs, detach=False, save_feat=False):
        # xs in the format [spec_1024, spec_512, spec_256, spec_128...]
        outs = []
        for i, disc in enumerate(self.discriminators):
            if detach:
                outs += [disc(xs[i].detach(), save_feat)]
            else:
                outs += [disc(xs[i], save_feat)]
        return outs





    
    
    