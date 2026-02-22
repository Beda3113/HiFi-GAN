"""Генератор HiFi-GAN"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


def _pad(k: int, d: int) -> int:
    """Вычисление padding для dilated convolution"""
    return (k * d - d) // 2


def _init_weights(m, std=0.01):
    """Инициализация весов"""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MRFBlock(nn.Module):
    """Multi-Receptive Field Fusion блок"""
    
    def __init__(self, channels, kernel_sizes, dilation_rates):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, k, dilation=d, padding=_pad(k, d)))
            for k in kernel_sizes for d in dilation_rates
        ])
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        for conv in self.convs:
            x = x + self.act(conv(x))
        return x


class Generator(nn.Module):
    """Генератор HiFi-GAN"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        n_mels = config['n_mels']
        hidden_dim = config['hidden_dim']
        upsample_kernels = config['upsample_kernel_sizes']
        kernel_sizes = config['kernel_sizes']
        dilation_rates = config['dilation_rates']
        upsample_rates = config['upsample_rates']
        
        self.enc_conv = weight_norm(nn.Conv1d(n_mels, hidden_dim, 7, padding=3))
        _init_weights(self.enc_conv)
        self.act = nn.LeakyReLU(0.1)
        
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        for i, (up_rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernels)):
            in_ch = hidden_dim // (2 ** i)
            out_ch = hidden_dim // (2 ** (i + 1))
            
            up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=up_rate, padding=kernel_size//2)
            _init_weights(up)
            self.ups.append(weight_norm(up))
            self.mrfs.append(MRFBlock(out_ch, kernel_sizes, dilation_rates))
        
        self.final_conv = weight_norm(nn.Conv1d(out_ch, 1, 7, padding='same'))
        _init_weights(self.final_conv)
    
    def forward(self, mel):
        """Генерация аудио из мел-спектрограммы
        
        Args:
            mel: [B, n_mels, T_mel]
            
        Returns:
            audio: [B, 1, T_audio]
        """
        x = self.act(self.enc_conv(mel))
        
        for up, mrf in zip(self.ups, self.mrfs):
            x = mrf(up(x))
            x = self.act(x)
        
        fake_waveform = torch.tanh(self.final_conv(x))
        return fake_waveform