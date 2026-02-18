import torch
import torch.nn as nn
import torch.nn.functional as F


class HiFiGANLoss(nn.Module):
    """Loss functions for HiFi-GAN training"""
    def __init__(self, lambda_fm=2.0, lambda_mel=45.0):
        super(HiFiGANLoss, self).__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def discriminator_loss(self, real_outputs, fake_outputs):
        """
        Calculate discriminator loss (LSGAN)
        
        Args:
            real_outputs: list of discriminator outputs for real samples
            fake_outputs: list of discriminator outputs for generated samples
        """
        loss = 0.0
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            loss += self.mse_loss(real_out, torch.ones_like(real_out)) + \
                    self.mse_loss(fake_out, torch.zeros_like(fake_out))
        return loss
    
    def generator_loss(self, fake_outputs):
        """
        Calculate generator adversarial loss (LSGAN)
        
        Args:
            fake_outputs: list of discriminator outputs for generated samples
        """
        loss = 0.0
        for fake_out in fake_outputs:
            loss += self.mse_loss(fake_out, torch.ones_like(fake_out))
        return loss
    
    def feature_matching_loss(self, real_features, fake_features):
        """
        Calculate feature matching loss
        
        Args:
            real_features: list of feature maps from real samples
            fake_features: list of feature maps from generated samples
        """
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            for r, f in zip(real_feat, fake_feat):
                loss += self.l1_loss(r, f)
        return loss * self.lambda_fm
    
    def mel_spectrogram_loss(self, mel_real, mel_fake):
        """Calculate mel-spectrogram L1 loss"""
        return self.l1_loss(mel_real, mel_fake) * self.lambda_mel


def feature_loss(fmap_r, fmap_g):
    """Feature matching loss for discriminators"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """Discriminator LSGAN loss"""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """Generator LSGAN loss"""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses