#!/usr/bin/env python
"""Скрипт для генерации аудио из текста"""

import argparse
import sys
from pathlib import Path

# Добавление родительской директории в путь
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
from tqdm import tqdm

from config.model_config import mel_config, generator_config
from models.generator import Generator
from models.mel_spectrogram import MelSpectrogram
from training.utils import get_device, save_audio_sample, ensure_dir


def generate_from_text(text, generator, mel_extractor, device, 
                      target_sr=22050, duration=2.0):
    """Генерация аудио из текста (заглушка)"""
    
    # В реальном проекте здесь должен быть text-to-mel encoder
    # Пока создаем случайную мел-спектрограмму как заглушку
    
    n_mels = mel_config['n_mels']
    hop_length = mel_config['hop_length']
    
    # Длительность в кадрах
    n_frames = int(duration * target_sr / hop_length)
    
    # Случайная мел-спектрограмма (заглушка)
    random_mel = torch.randn(1, n_mels, n_frames).to(device)
    
    with torch.no_grad():
        audio = generator(random_mel)
    
    return audio.squeeze().cpu().numpy()


def generate_from_mel(mel_path, generator, mel_extractor, device, output_path):
    """Генерация аудио из сохраненной мел-спектрограммы"""
    
    mel = np.load(mel_path)
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)
    
    with torch.no_grad():
        audio = generator(mel)
    
    audio_np = audio.squeeze().cpu().numpy()
    sr = mel_config['sample_rate']
    
    save_audio_sample(output_path, audio_np, sr)
    print(f" Аудио сохранено: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Генерация аудио')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Путь к чекпоинту генератора')
    parser.add_argument('--input', type=str, 
                       help='Входной файл (текст или .npy мел-спектрограмма)')
    parser.add_argument('--output_dir', type=str, default='./generated',
                       help='Директория для сохранения')
    parser.add_argument('--text', type=str, default=None,
                       help='Текст для генерации')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Количество сэмплов для генерации')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Длительность в секундах')
    
    args = parser.parse_args()
    
    # Устройство
    device = get_device()
    
    # Модели
    generator = Generator(generator_config).to(device)
    mel_extractor = MelSpectrogram(mel_config).to(device)
    
    # Загрузка чекпоинта
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        generator.load_state_dict(checkpoint)
    
    generator.eval()
    print(f" Генератор загружен из {args.checkpoint}")
    
    # Создание выходной директории
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.input and args.input.endswith('.npy'):
        # Генерация из мел-спектрограммы
        out_path = output_dir / f"generated_{Path(args.input).stem}.wav"
        generate_from_mel(args.input, generator, mel_extractor, device, out_path)
    
    elif args.text or args.input:
        # Генерация из текста
        text = args.text or Path(args.input).read_text(encoding='utf-8').strip()
        
        print(f"\n Текст: {text}")
        
        audio = generate_from_text(
            text, generator, mel_extractor, device,
            duration=args.duration
        )
        
        out_path = output_dir / f"generated_text.wav"
        sr = mel_config['sample_rate']
        save_audio_sample(out_path, audio, sr)
        
        print(f" Аудио сохранено: {out_path}")
    
    else:
        # Генерация случайных сэмплов
        print(f"\n Генерация {args.num_samples} случайных сэмплов...")
        
        for i in tqdm(range(args.num_samples)):
            audio = generate_from_text(
                f"Sample {i}", generator, mel_extractor, device,
                duration=args.duration
            )
            
            out_path = output_dir / f"sample_{i:03d}.wav"
            sr = mel_config['sample_rate']
            save_audio_sample(out_path, audio, sr)
        
        print(f"\n {args.num_samples} сэмплов сохранено в {output_dir}")


if __name__ == '__main__':
    main()