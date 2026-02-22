#!/usr/bin/env python
"""Скрипт для обучения HiFi-GAN"""

import argparse
import sys
from pathlib import Path

# Добавление родительской директории в путь
sys.path.append(str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from config.model_config import mel_config, generator_config
from config.training_config import training_config, data_config
from data.dataset import RUSLANDataset
from models.generator import Generator
from models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.mel_spectrogram import MelSpectrogram
from models.loss import HiFiGANLoss
from training.trainer import HiFiGANTrainer


def main():
    parser = argparse.ArgumentParser(description='Обучение HiFi-GAN')
    parser.add_argument('--data_dir', type=str, default='/kaggle/working/data/ruslan',
                       help='Директория с данными')
    parser.add_argument('--checkpoint_dir', type=str, default='/kaggle/working/checkpoints',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/output',
                       help='Директория для выходных файлов')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Количество эпох')
    parser.add_argument('--resume', type=str, default=None,
                       help='Путь к чекпоинту для продолжения обучения')
    
    args = parser.parse_args()
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")
    
    # Датасет
    index_path = Path(args.data_dir) / 'custom_dir_index.json'
    
    train_dataset = RUSLANDataset(
        index_path, 
        data_config, 
        part='train', 
        augment=True
    )
    val_dataset = RUSLANDataset(
        index_path, 
        data_config, 
        part='val', 
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Модели
    generator = Generator(generator_config).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    mel_extractor = MelSpectrogram(mel_config).to(device)
    
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"MPD params: {sum(p.numel() for p in mpd.parameters()):,}")
    print(f"MSD params: {sum(p.numel() for p in msd.parameters()):,}")
    
    # Loss
    criterion = HiFiGANLoss(
        lambda_fm=training_config['lambda_fm'],
        lambda_mel=training_config['lambda_mel']
    ).to(device)
    
    # Оптимизаторы
    optimizer_g = torch.optim.AdamW(
        generator.parameters(),
        lr=training_config['lr_generator'],
        betas=(training_config['beta1'], training_config['beta2']),
        weight_decay=training_config['weight_decay']
    )
    
    optimizer_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=training_config['lr_discriminator'],
        betas=(training_config['beta1'], training_config['beta2']),
        weight_decay=training_config['weight_decay']
    )
    
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.999)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.999)
    
    # Тренер
    trainer = HiFiGANTrainer(
        generator=generator,
        mpd=mpd,
        msd=msd,
        mel_extractor=mel_extractor,
        criterion=criterion,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        scheduler_g=scheduler_g,
        scheduler_d=scheduler_d,
        device=device,
        config=training_config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
    
    # Загрузка чекпоинта
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Обучение
    trainer.train(train_loader, val_loader, args.epochs, start_epoch)


if __name__ == '__main__':
    main()