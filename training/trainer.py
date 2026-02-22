"""Класс для обучения HiFi-GAN"""

import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import soundfile as sf
from tqdm import tqdm


class HiFiGANTrainer:
    """Тренер для HiFi-GAN"""
    
    def __init__(self, generator, mpd, msd, mel_extractor, criterion,
                 optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                 device, config, checkpoint_dir, output_dir):
        
        self.generator = generator
        self.mpd = mpd
        self.msd = msd
        self.mel_extractor = mel_extractor
        self.criterion = criterion
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.device = device
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.output_dir = Path(output_dir)
        
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir.parent / 'tensorboard'))
        self.loss_history = []
        self.best_g_loss = float('inf')
    
    def train_epoch(self, train_loader, epoch):
        """Обучение одной эпохи"""
        
        self.generator.train()
        self.mpd.train()
        self.msd.train()
        
        losses = {'d': 0, 'g': 0, 'g_adv': 0, 'g_fm': 0, 'g_mel': 0}
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        self.optimizer_d.zero_grad()
        self.optimizer_g.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            try:
                audio = batch['audio'].to(self.device)
                
                with torch.no_grad():
                    real_mel = self.mel_extractor(audio)
                
                fake_audio = self.generator(real_mel)
                fake_mel = self.mel_extractor(fake_audio)
                
                # Discriminator step
                fake_audio_detach = fake_audio.detach()
                fake_feats_mpd = self.mpd(fake_audio_detach)
                real_feats_mpd = self.mpd(audio)
                fake_feats_msd = self.msd(fake_audio_detach)
                real_feats_msd = self.msd(audio)
                
                d_loss = self.criterion.discriminator_loss(
                    fake_feats_mpd + fake_feats_msd,
                    real_feats_mpd + real_feats_msd
                )
                d_loss = d_loss / self.config['accumulation_steps']
                d_loss.backward()
                
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    clip_grad_norm_(
                        list(self.mpd.parameters()) + list(self.msd.parameters()),
                        self.config['gradient_clip']
                    )
                    self.optimizer_d.step()
                    self.optimizer_d.zero_grad()
                
                # Generator step
                fake_feats_mpd = self.mpd(fake_audio)
                real_feats_mpd = self.mpd(audio)
                fake_feats_msd = self.msd(fake_audio)
                real_feats_msd = self.msd(audio)
                
                # Выравнивание мел-спектрограмм
                min_T = min(real_mel.shape[-1], fake_mel.shape[-1])
                real_mel_aligned = real_mel[..., :min_T]
                fake_mel_aligned = fake_mel[..., :min_T]
                
                g_losses = self.criterion.generator_loss(
                    fake_feats_mpd + fake_feats_msd,
                    real_feats_mpd + real_feats_msd,
                    fake_mel_aligned,
                    real_mel_aligned
                )
                g_loss = g_losses['total'] / self.config['accumulation_steps']
                g_loss.backward()
                
                if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                    clip_grad_norm_(
                        self.generator.parameters(),
                        self.config['gradient_clip']
                    )
                    self.optimizer_g.step()
                    self.optimizer_g.zero_grad()
                
                # Статистика
                losses['d'] += d_loss.item() * self.config['accumulation_steps']
                losses['g'] += g_losses['total'].item()
                losses['g_adv'] += g_losses['adv'].item()
                losses['g_fm'] += g_losses['fm'].item()
                losses['g_mel'] += g_losses['mel'].item()
                batch_count += 1
                
                pbar.set_postfix({
                    'D': f"{d_loss.item() * self.config['accumulation_steps']:.3f}",
                    'G': f"{g_losses['total'].item():.3f}"
                })
                
            except Exception as e:
                print(f"\nОшибка в батче {batch_idx}: {e}")
                continue
        
        for k in losses:
            losses[k] /= batch_count
        
        return losses
    
    def validate(self, val_loader):
        """Валидация модели"""
        
        self.generator.eval()
        self.mpd.eval()
        self.msd.eval()
        
        losses = {'d': 0, 'g_mel': 0}
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(self.device)
                real_mel = self.mel_extractor(audio)
                fake_audio = self.generator(real_mel)
                fake_mel = self.mel_extractor(fake_audio)
                
                fake_feats_mpd = self.mpd(fake_audio)
                real_feats_mpd = self.mpd(audio)
                fake_feats_msd = self.msd(fake_audio)
                real_feats_msd = self.msd(audio)
                
                d_loss = self.criterion.discriminator_loss(
                    fake_feats_mpd + fake_feats_msd,
                    real_feats_mpd + real_feats_msd
                )
                
                # Выравнивание мел-спектрограмм
                min_T = min(real_mel.shape[-1], fake_mel.shape[-1])
                g_mel_loss = self.criterion.mel_loss(
                    fake_mel[..., :min_T],
                    real_mel[..., :min_T]
                )
                
                losses['d'] += d_loss.item()
                losses['g_mel'] += g_mel_loss.item()
                batch_count += 1
                
                if batch_count >= 20:
                    break
        
        for k in losses:
            losses[k] /= batch_count
        
        return losses
    
    def save_samples(self, val_loader, epoch):
        """Сохранение примеров синтеза"""
        
        self.generator.eval()
        sample_dir = self.output_dir / f'epoch_{epoch:03d}'
        sample_dir.mkdir(exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 3:
                    break
                
                audio = batch['audio'][:1].to(self.device)
                mel = self.mel_extractor(audio)
                fake = self.generator(mel)
                
                # Сохранение в WAV
                audio_np = audio[0].cpu().numpy()
                fake_np = fake[0].cpu().numpy()
                
                if audio_np.ndim > 1:
                    audio_np = audio_np.squeeze()
                if fake_np.ndim > 1:
                    fake_np = fake_np.squeeze()
                
                audio_np = np.clip(audio_np, -1, 1)
                fake_np = np.clip(fake_np, -1, 1)
                
                sr = self.mel_extractor.sr
                
                sf.write(sample_dir / f'{i}_real.wav', audio_np, sr)
                sf.write(sample_dir / f'{i}_fake.wav', fake_np, sr)
                
                self.writer.add_audio(f'Real_{i}', audio[0], epoch, sr)
                
                fake_for_tb = fake[0].squeeze() if fake[0].dim() > 1 else fake[0]
                self.writer.add_audio(f'Fake_{i}', fake_for_tb, epoch, sr)
        
        print(f"Сэмплы сохранены: {sample_dir}")
    
    def save_checkpoint(self, epoch, is_best=False):
        """Сохранение чекпоинта"""
        
        checkpoint = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'mpd': self.mpd.state_dict(),
            'msd': self.msd.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'scheduler_g': self.scheduler_g.state_dict(),
            'scheduler_d': self.scheduler_d.state_dict(),
            'loss_history': self.loss_history,
        }
        
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_{epoch:03d}.pt')
        
        if is_best:
            torch.save(self.generator.state_dict(), self.checkpoint_dir / 'generator_best.pt')
    
    def load_checkpoint(self, path):
        """Загрузка чекпоинта"""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.mpd.load_state_dict(checkpoint['mpd'])
        self.msd.load_state_dict(checkpoint['msd'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d'])
        self.loss_history = checkpoint['loss_history']
        
        print(f"Загружен чекпоинт эпохи {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader, num_epochs, start_epoch=1):
        """Основной цикл обучения"""
        
        print("\n" + "="*60)
        print("🚀 ЗАПУСК ОБУЧЕНИЯ")
        print("="*60)
        
        training_start = datetime.now()
        
        for epoch in range(start_epoch, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"ЭПОХА {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Обучение
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Логирование
            self.writer.add_scalar('Loss/Train_D', train_losses['d'], epoch)
            self.writer.add_scalar('Loss/Train_G', train_losses['g'], epoch)
            self.writer.add_scalar('Loss/Train_Mel', train_losses['g_mel'], epoch)
            self.writer.add_scalar('Params/LR_G', self.optimizer_g.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('Params/LR_D', self.optimizer_d.param_groups[0]['lr'], epoch)
            
            print(f"\n📈 TRAIN:")
            print(f"  D: {train_losses['d']:.4f}")
            print(f"  G: {train_losses['g']:.4f}")
            print(f"  Mel: {train_losses['g_mel']:.4f}")
            
            # Валидация и сохранение
            if epoch % self.config['save_interval'] == 0:
                val_losses = self.validate(val_loader)
                
                self.writer.add_scalar('Loss/Val_D', val_losses['d'], epoch)
                self.writer.add_scalar('Loss/Val_Mel', val_losses['g_mel'], epoch)
                
                print(f"\n📊 VAL:")
                print(f"  D: {val_losses['d']:.4f}")
                print(f"  Mel: {val_losses['g_mel']:.4f}")
                
                is_best = val_losses['g_mel'] < self.best_g_loss
                if is_best:
                    self.best_g_loss = val_losses['g_mel']
                    print(f"  ⭐ Лучшая модель!")
                
                self.save_samples(val_loader, epoch)
                self.save_checkpoint(epoch, is_best)
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            self.loss_history.append(train_losses)
        
        # Итоги
        training_end = datetime.now()
        duration = training_end - training_start
        
        print(f"\n{'='*60}")
        print("ИТОГИ ОБУЧЕНИЯ")
        print(f"{'='*60}")
        print(f"Длительность: {duration}")
        print(f"Завершено эпох: {len(self.loss_history)}")
        print(f"Лучший G Mel Loss: {self.best_g_loss:.4f}")
        
        torch.save(self.generator.state_dict(), self.checkpoint_dir / 'generator_final.pt')
        self.writer.close()