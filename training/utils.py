"""Утилиты для обучения"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path


def count_parameters(model):
    """Подсчет количества обучаемых параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_audio_sample(path, audio, sr):
    """Сохранение аудио с правильной обработкой размерностей"""
    
    if torch.is_tensor(audio):
        audio = audio.detach().cpu().numpy()
    
    # Убираем лишние размерности
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Клиппинг в [-1, 1]
    audio = np.clip(audio, -1, 1)
    
    sf.write(path, audio, sr, format='WAV', subtype='PCM_16')
    return path


def plot_spectrogram(mel, path=None, title="Mel-spectrogram"):
    """Визуализация мел-спектрограммы"""
    
    if torch.is_tensor(mel):
        mel = mel.detach().cpu().numpy()
    
    if mel.ndim > 2:
        mel = mel[0]  # берем первый из батча
    
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time frames')
    plt.ylabel('Mel bins')
    plt.tight_layout()
    
    if path:
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """Загрузка чекпоинта"""
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['generator'])
    
    if optimizer and 'optimizer_g' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_g'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss_history', [])
    
    print(f" Загружен чекпоинт: эпоха {epoch}")
    
    return {
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'loss_history': loss
    }


def get_device():
    """Получение доступного устройства"""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f" CUDA доступна: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(" CUDA не найдена, используется CPU")
    
    return device


def save_checkpoint(generator, mpd, msd, optimizer_g, optimizer_d, 
                   epoch, loss, path):
    """Сохранение полного чекпоинта"""
    
    checkpoint = {
        'epoch': epoch,
        'generator': generator.state_dict(),
        'mpd': mpd.state_dict(),
        'msd': msd.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_d': optimizer_d.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, path)
    print(f" Чекпоинт сохранен: {path}")


def ensure_dir(path):
    """Создание директории если не существует"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path