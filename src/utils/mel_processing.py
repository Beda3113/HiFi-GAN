import torch
import torch.nn.functional as F
import numpy as np
from librosa.filters import mel as librosa_mel_fn

# Global cache for mel basis and windows
mel_basis = {}
hann_window = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """Dynamic range compression for spectrogram"""
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """Dynamic range decompression for spectrogram"""
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    """Normalize spectrogram"""
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    """Denormalize spectrogram"""
    return dynamic_range_decompression_torch(magnitudes)


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """Compute mel-spectrogram from waveform"""
    if torch.min(y) < -1.0:
        print(f"Warning: min value is {torch.min(y)}")
    if torch.max(y) > 1.0:
        print(f"Warning: max value is {torch.max(y)}")

    global mel_basis, hann_window
    
    # Create unique key for mel basis
    fmax_key = f"{fmax}_{y.device}"
    if fmax_key not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_key] = torch.from_numpy(mel).float().to(y.device)
    
    # Create unique key for hann window
    window_key = str(y.device)
    if window_key not in hann_window:
        hann_window[window_key] = torch.hann_window(win_size).to(y.device)

    # Padding for STFT
    y = F.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[window_key],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    
    spec = torch.abs(spec)
    spec = torch.matmul(mel_basis[fmax_key], spec)
    spec = spectral_normalize_torch(spec)
    
    return spec


class MelSpectrogramExtractor:
    """Wrapper class for mel-spectrogram extraction"""
    def __init__(self, config):
        self.n_fft = config["n_fft"]
        self.num_mels = config["n_mels"]
        self.sampling_rate = config["sampling_rate"]
        self.hop_size = config["hop_length"]
        self.win_size = config["win_length"]
        self.fmin = config["f_min"]
        self.fmax = config["f_max"]
    
    def __call__(self, audio):
        return mel_spectrogram(
            audio, self.n_fft, self.num_mels, self.sampling_rate,
            self.hop_size, self.win_size, self.fmin, self.fmax, center=False
        )