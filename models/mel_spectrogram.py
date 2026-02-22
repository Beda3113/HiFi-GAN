"""Модуль для вычисления мел-спектрограмм"""

import torch
import torch.nn as nn
import librosa
import numpy as np


class MelSpectrogram(nn.Module):
    """Извлечение мел-спектрограммы из аудио"""
    
    def __init__(self, config):
        super().__init__()
        self.sr = config['sample_rate']
        self.n_fft = config['n_fft']
        self.hop_length = config['hop_length']
        self.win_length = config['win_length']
        self.n_mels = config['n_mels']
        self.f_min = config['f_min']
        self.f_max = config['f_max']
        
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        self.register_buffer('hann_window', torch.hann_window(self.win_length).float())
    
    def forward(self, audio):
        """Преобразование аудио в мел-спектрограмму
        
        Args:
            audio: [B, T] или [B, 1, T]
            
        Returns:
            mel: [B, n_mels, T']
        """
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            return_complex=True
        )
        mag = torch.abs(spec)
        mel = torch.matmul(self.mel_basis, mag)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel