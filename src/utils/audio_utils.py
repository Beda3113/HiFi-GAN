import torch
import numpy as np
import soundfile as sf
import librosa
from scipy.io.wavfile import write
import os

MAX_WAV_VALUE = 32768.0


def load_wav(full_path, target_sr=22050):
    """Load wav file and resample if necessary"""
    audio, sr = sf.read(full_path)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if necessary
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    return audio, sr


def save_wav(wav, path, sr=22050):
    """Save wav file"""
    wav = wav * MAX_WAV_VALUE
    wav = wav.astype('int16')
    write(path, sr, wav)


def normalize_audio(audio):
    """Normalize audio to [-1, 1]"""
    return audio / MAX_WAV_VALUE


def denormalize_audio(audio):
    """Denormalize audio from [-1, 1] to int16 range"""
    return (audio * MAX_WAV_VALUE).astype('int16')