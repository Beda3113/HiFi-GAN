import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator


class HiFiGAN(nn.Module):
    """Complete HiFi-GAN model with generator and discriminators"""
    def __init__(self, config):
        super(HiFiGAN, self).__init__()
        
        self.generator = Generator(config)
        self.mpd = MultiPeriodDiscriminator(config.get("periods", [2, 3, 5, 7, 11]))
        self.msd = MultiScaleDiscriminator()
        
    def forward(self, mel):
        """Generate audio from mel-spectrogram"""
        return self.generator(mel)
    
    def discriminate(self, y, y_hat):
        """Run discriminators on real and fake samples"""
        mpd_outputs = self.mpd(y, y_hat)
        msd_outputs = self.msd(y, y_hat)
        return mpd_outputs, msd_outputs