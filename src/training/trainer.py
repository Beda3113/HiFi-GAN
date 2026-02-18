import os
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
import wandb

from ..utils.utils import scan_checkpoint, load_checkpoint, save_checkpoint, plot_spectrogram
from ..utils.mel_processing import mel_spectrogram
from ..models.hifigan import HiFiGAN
from .losses import discriminator_loss, generator_loss, feature_loss


class HiFiGANTrainer:
    """Trainer for HiFi-GAN"""
    def __init__(self, config, rank=0):
        self.config = config
        self.rank = rank
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{rank}')
            torch.cuda.manual_seed(config["seed"])
        else:
            self.device = torch.device('cpu')
        
        # Create model
        self.model = HiFiGAN(config).to(self.device)
        
        # Setup optimizers
        self.optim_g = torch.optim.AdamW(
            self.model.generator.parameters(),
            lr=config["learning_rate"],
            betas=[config["adam_b1"], config["adam_b2"]]
        )
        
        self.optim_d = torch.optim.AdamW(
            list(self.model.mpd.parameters()) + list(self.model.msd.parameters()),
            lr=config["learning_rate"],
            betas=[config["adam_b1"], config["adam_b2"]]
        )
        
        # Setup schedulers
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=config["lr_decay"]
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=config["lr_decay"]
        )
        
        # Tracking
        self.steps = 0
        self.epoch = 0
        
        # Initialize wandb (if rank 0)
        if rank == 0 and config.get("use_wandb", False):
            wandb.init(project="hifigan-ruslan", config=config)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        state_dict_g = load_checkpoint(checkpoint_path, self.device)
        self.model.generator.load_state_dict(state_dict_g['generator'])
        
        if 'mpd' in state_dict_g and 'msd' in state_dict_g:
            self.model.mpd.load_state_dict(state_dict_g['mpd'])
            self.model.msd.load_state_dict(state_dict_g['msd'])
        
        if 'optim_g' in state_dict_g and 'optim_d' in state_dict_g:
            self.optim_g.load_state_dict(state_dict_g['optim_g'])
            self.optim_d.load_state_dict(state_dict_g['optim_d'])
            self.steps = state_dict_g.get('steps', 0)
            self.epoch = state_dict_g.get('epoch', 0)
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        save_checkpoint(path, {
            'generator': self.model.generator.state_dict(),
            'mpd': self.model.mpd.state_dict(),
            'msd': self.model.msd.state_dict(),
            'optim_g': self.optim_g.state_dict(),
            'optim_d': self.optim_d.state_dict(),
            'steps': self.steps,
            'epoch': self.epoch,
            'config': self.config
        })
    
    def train_step(self, batch):
        """Single training step"""
        x, y, _, y_mel = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        y_mel = y_mel.to(self.device, non_blocking=True)
        y = y.unsqueeze(1)
        
        # Generate audio
        y_g_hat = self.model.generator(x)
        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1),
            self.config["n_fft"],
            self.config["num_mels"],
            self.config["sampling_rate"],
            self.config["hop_size"],
            self.config["win_size"],
            self.config["fmin"],
            self.config["fmax_for_loss"] or self.config["fmax"],
            center=False
        )
        
        # Train Discriminators
        self.optim_d.zero_grad()
        
        # MPD
        mpd_outputs = self.model.mpd(y, y_g_hat.detach())
        mpd_y_d_rs, mpd_y_d_gs, _, _ = mpd_outputs
        
        # MSD
        msd_outputs = self.model.msd(y, y_g_hat.detach())
        msd_y_d_rs, msd_y_d_gs, _, _ = msd_outputs
        
        # Discriminator losses
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(mpd_y_d_rs, mpd_y_d_gs)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(msd_y_d_rs, msd_y_d_gs)
        loss_disc_all = loss_disc_s + loss_disc_f
        
        loss_disc_all.backward()
        self.optim_d.step()
        
        # Train Generator
        self.optim_g.zero_grad()
        
        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * self.config["lambda_mel"]
        
        # Get discriminator outputs for generator
        mpd_outputs = self.model.mpd(y, y_g_hat)
        mpd_y_d_rs, mpd_y_d_gs, mpd_fmap_rs, mpd_fmap_gs = mpd_outputs
        
        msd_outputs = self.model.msd(y, y_g_hat)
        msd_y_d_rs, msd_y_d_gs, msd_fmap_rs, msd_fmap_gs = msd_outputs
        
        # Feature matching loss
        loss_fm_f = feature_loss(mpd_fmap_rs, mpd_fmap_gs)
        loss_fm_s = feature_loss(msd_fmap_rs, msd_fmap_gs)
        
        # Generator adversarial loss
        loss_gen_f, _ = generator_loss(mpd_y_d_gs)
        loss_gen_s, _ = generator_loss(msd_y_d_gs)
        
        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        
        loss_gen_all.backward()
        self.optim_g.step()
        
        return {
            "loss_disc": loss_disc_all.item(),
            "loss_gen": loss_gen_all.item(),
            "loss_mel": loss_mel.item() / self.config["lambda_mel"],
            "loss_fm": (loss_fm_s + loss_fm_f).item() / self.config["lambda_fm"],
        }
    
    def validate(self, val_loader, writer):
        """Run validation"""
        self.model.generator.eval()
        val_err = 0
        
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                x, y, _, y_mel = batch
                x = x.to(self.device)
                y = y.to(self.device)
                y_mel = y_mel.to(self.device)
                
                y_g_hat = self.model.generator(x)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    self.config["n_fft"],
                    self.config["num_mels"],
                    self.config["sampling_rate"],
                    self.config["hop_size"],
                    self.config["win_size"],
                    self.config["fmin"],
                    self.config["fmax_for_loss"] or self.config["fmax"],
                    center=False
                )
                
                val_err += F.l1_loss(y_mel, y_g_hat_mel).item()
                
                # Log audio samples
                if j < 4 and self.rank == 0:
                    writer.add_audio(
                        f'generated/sample_{j}',
                        y_g_hat[0].cpu(),
                        self.steps,
                        self.config["sampling_rate"]
                    )
        
        val_err = val_err / (j + 1)
        
        if self.rank == 0:
            writer.add_scalar("validation/mel_error", val_err, self.steps)
            if self.config.get("use_wandb", False):
                wandb.log({"val_mel_error": val_err}, step=self.steps)
        
        self.model.generator.train()
        return val_err