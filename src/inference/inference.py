import os
import torch
import numpy as np
from scipy.io.wavfile import write
from ..utils.audio_utils import MAX_WAV_VALUE
from ..utils.mel_processing import mel_spectrogram
from ..models.generator import Generator


class HiFiGANInference:
    """Inference wrapper for HiFi-GAN"""
    def __init__(self, checkpoint_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # Load generator
        self.generator = Generator(config).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        if 'generator' in state_dict:
            self.generator.load_state_dict(state_dict['generator'])
        else:
            self.generator.load_state_dict(state_dict)
        
        self.generator.eval()
        self.generator.remove_weight_norm()
    
    def mel_to_wav(self, mel):
        """
        Convert mel-spectrogram to waveform
        
        Args:
            mel: [C, T] or [1, C, T] numpy array or torch tensor
        Returns:
            audio: numpy array [T']
        """
        with torch.no_grad():
            if isinstance(mel, np.ndarray):
                mel = torch.FloatTensor(mel)
            
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            
            mel = mel.to(self.device)
            audio = self.generator(mel)
            audio = audio.squeeze().cpu().numpy()
            audio = audio * MAX_WAV_VALUE
            audio = audio.astype('int16')
        
        return audio
    
    def audio_to_mel(self, audio):
        """Extract mel-spectrogram from audio"""
        if isinstance(audio, np.ndarray):
            audio = torch.FloatTensor(audio).unsqueeze(0)
        
        audio = audio.to(self.device)
        mel = mel_spectrogram(
            audio,
            self.config["n_fft"],
            self.config["num_mels"],
            self.config["sampling_rate"],
            self.config["hop_size"],
            self.config["win_size"],
            self.config["fmin"],
            self.config["fmax"],
            center=False
        )
        
        return mel.squeeze(0).cpu().numpy()