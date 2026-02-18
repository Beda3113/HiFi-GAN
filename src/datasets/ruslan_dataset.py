import os
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from ..utils.audio_utils import load_wav, normalize_audio, MAX_WAV_VALUE
from ..utils.mel_processing import mel_spectrogram


class RuslanDataset(Dataset):
    """Dataset for RUSLAN (or LJSpeech) training"""
    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
    ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
    
    def __getitem__(self, index):
        filename = self.audio_files[index]
        
        # Load audio with caching
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename, self.sampling_rate)
            audio = audio / MAX_WAV_VALUE
            if not self.fine_tuning:
                audio = audio * 0.95  # Slight normalization
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1
        
        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        
        # Handle segmentation
        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_start = audio.size(1) - self.segment_size
                    start = random.randint(0, max_start)
                    audio = audio[:, start:start + self.segment_size]
                else:
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)), "constant")
            
            # Extract mel-spectrogram
            mel = mel_spectrogram(
                audio, self.n_fft, self.num_mels, self.sampling_rate,
                self.hop_size, self.win_size, self.fmin, self.fmax, center=False
            )
        else:
            # Fine-tuning mode: load pre-computed mel
            mel = np.load(os.path.join(
                self.base_mels_path,
                os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'
            ))
            mel = torch.from_numpy(mel)
            
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)
            
            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)
                
                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = F.pad(mel, (0, frames_per_seg - mel.size(2)), "constant")
                    audio = F.pad(audio, (0, self.segment_size - audio.size(1)), "constant")
        
        # Extract mel for loss computation
        mel_loss = mel_spectrogram(
            audio, self.n_fft, self.num_mels, self.sampling_rate,
            self.hop_size, self.win_size, self.fmin, self.fmax_loss or self.fmax, center=False
        )
        
        return mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze()
    
    def __len__(self):
        return len(self.audio_files)


def get_dataset_filelist(input_training_file, input_validation_file, input_wavs_dir):
    """Get file lists for training and validation"""
    with open(input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [
            os.path.join(input_wavs_dir, x.split('|')[0] + '.wav')
            for x in fi.read().split('\n') if len(x) > 0
        ]
    
    with open(input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [
            os.path.join(input_wavs_dir, x.split('|')[0] + '.wav')
            for x in fi.read().split('\n') if len(x) > 0
        ]
    
    return training_files, validation_files