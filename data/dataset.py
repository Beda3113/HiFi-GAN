"""Dataset класс для RUSLAN"""

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import soundfile as sf


class RUSLANDataset(Dataset):
    """
    Dataset с аугментацией для увеличения эффективного размера
    """
    
    def __init__(self, index_path, config, part='train', augment=True):
        self.config = config
        self.sr = config['sample_rate']
        self.segment_size = config['segment_size']
        self.part = part
        self.augment = augment and (part == 'train')
        
        # Загрузка индекса
        with open(index_path, 'r', encoding='utf-8') as f:
            self.index = json.load(f)
        
        # Разделение train/val
        random.seed(42)
        indices = list(range(len(self.index)))
        random.shuffle(indices)
        split_idx = int(0.9 * len(indices))
        
        if part == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
    
    def __len__(self):
        return len(self.indices)
    
    def _load_audio(self, path):
        try:
            audio, sr = sf.read(path)
            audio = audio.astype(np.float32)
            
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            
            if sr != self.sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
            
            return audio
        except Exception as e:
            print(f"Ошибка загрузки {path}: {e}")
            return None
    
    def _augment(self, audio):
        """Аугментация для увеличения разнообразия"""
        if not self.augment:
            return audio
        
        aug_type = np.random.randint(0, 4)
        
        if aug_type == 0:
            # Изменение скорости
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        elif aug_type == 1:
            # Изменение питча
            steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=steps)
        elif aug_type == 2:
            # Добавление шума
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        elif aug_type == 3:
            # Изменение громкости
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
        
        return audio
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.index[real_idx]
        
        audio = self._load_audio(item['path'])
        
        if audio is None or len(audio) < 1000:
            audio = np.random.randn(self.segment_size).astype(np.float32) * 0.01
        else:
            audio = self._augment(audio)
            
            # Нормализация
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / (peak + 1e-7)
            audio = np.clip(audio, -1, 1)
        
        # Сэмплирование сегмента
        if len(audio) > self.segment_size:
            if self.part == 'train':
                start = np.random.randint(0, len(audio) - self.segment_size)
            else:
                start = 0
            audio = audio[start:start + self.segment_size]
        else:
            audio = np.pad(audio, (0, self.segment_size - len(audio)))
        
        return {
            'audio': torch.FloatTensor(audio),
            'text': item['text'],
            'path': item['path']
        }