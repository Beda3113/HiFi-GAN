"""Функции потерь для HiFi-GAN"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """LSGAN loss function"""
    
    def __init__(self, is_discriminator=True):
        super().__init__()
        self.is_disc = is_discriminator
    
    def forward(self, predictions, real_predictions=None):
        if self.is_disc:
            real_loss = torch.mean((real_predictions - 1.0) ** 2)
            fake_loss = torch.mean(predictions ** 2)
            return real_loss + fake_loss
        else:
            return torch.mean((predictions - 1.0) ** 2)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss"""
    
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
    
    def forward(self, fake_features, real_features):
        loss = 0
        for fake_feat_list, real_feat_list in zip(fake_features, real_features):
            for fake_feat, real_feat in zip(fake_feat_list, real_feat_list):
                min_len = min(fake_feat.shape[-1], real_feat.shape[-1])
                fake_feat = fake_feat[..., :min_len]
                real_feat = real_feat[..., :min_len]
                loss += self.l1(fake_feat, real_feat)
        return loss


class HiFiGANLoss(nn.Module):
    """Complete HiFi-GAN loss"""
    
    def __init__(self, lambda_fm=2.0, lambda_mel=45.0):
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        
        self.gan_loss_disc = GANLoss(is_discriminator=True)
        self.gan_loss_gen = GANLoss(is_discriminator=False)
        self.feature_loss = FeatureMatchingLoss()
        self.mel_loss = nn.L1Loss()
    
    def discriminator_loss(self, fake_features, real_features):
        total_loss = 0
        for fake_feats, real_feats in zip(fake_features, real_features):
            total_loss += self.gan_loss_disc(fake_feats[-1], real_feats[-1])
        return total_loss
    
    def generator_loss(self, fake_features, real_features, fake_mel, real_mel):
        # Adversarial loss
        adv_loss = 0
        for fake_feats, real_feats in zip(fake_features, real_features):
            adv_loss += self.gan_loss_gen(fake_feats[-1])
        
        # Feature matching loss
        fm_loss = self.feature_loss(fake_features, real_features)
        
        # Mel spectrogram loss
        mel_loss = self.mel_loss(fake_mel, real_mel)
        
        total_loss = adv_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_loss
        
        return {
            'total': total_loss,
            'adv': adv_loss,
            'fm': fm_loss,
            'mel': mel_loss
        }