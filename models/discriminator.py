"""Дискриминаторы HiFi-GAN (MPD и MSD)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class PeriodDiscriminator(nn.Module):
    """Периодический дискриминатор"""
    
    def __init__(self, period):
        super().__init__()
        self.period = period
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 64, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(64, 128, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), (2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), (1, 1), (2, 0))),
        ])
        self.act = nn.LeakyReLU(0.1)
        self.conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), (1, 1), (1, 0)))
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        B, C, T = x.shape
        pad = (self.period - T % self.period) % self.period
        if pad > 0:
            x = F.pad(x, (0, pad), mode='reflect')
            T += pad
        
        x = x.view(B, C, T // self.period, self.period)
        features = []
        
        for conv in self.convs:
            x = self.act(conv(x))
            features.append(x)
        
        x = self.conv(x)
        features.append(x)
        return features


class MultiPeriodDiscriminator(nn.Module):
    """Многопериодический дискриминатор"""
    
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(p) for p in periods
        ])
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = []
        for d in self.discriminators:
            features.append(d(x))
        return features


class ScaleDiscriminator(nn.Module):
    """Масштабный дискриминатор"""
    
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, 7)),
            norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.act = nn.LeakyReLU(0.1)
        self.conv = norm(nn.Conv1d(1024, 1, 3, 1, 1))
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = []
        for conv in self.convs:
            x = self.act(conv(x))
            features.append(x)
        x = self.conv(x)
        features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):
    """Многомасштабный дискриминатор"""
    
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(True),
            ScaleDiscriminator(False),
            ScaleDiscriminator(False),
        ])
        self.avgpools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2),
        ])
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        features = []
        for avg, d in zip(self.avgpools, self.discriminators):
            x = avg(x)
            features.append(d(x))
        return features