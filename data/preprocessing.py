"""Предобработка данных для RUSLAN датасета"""

import json
import shutil
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm


def create_index(audio_dir, text_dir, output_path):
    """Создание индексного файла из аудио и текстовых файлов"""
    
    audio_dir = Path(audio_dir)
    text_dir = Path(text_dir)
    
    index = []
    text_files = sorted(list(text_dir.glob('*.txt')))
    
    for txt_path in tqdm(text_files, desc="Индексация"):
        stem = txt_path.stem
        wav_path = audio_dir / f'{stem}.wav'
        
        if not wav_path.exists():
            continue
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        index.append({
            'path': str(wav_path.resolve()),
            'text': text,
            'has_audio': True
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f" Создан индекс: {len(index)} записей")
    return index


def extract_metadata_from_csv(csv_path, src_dir, dst_audio_dir, dst_text_dir, limit=None):
    """Извлечение данных из CSV файла метаданных"""
    
    wavs_copied = 0
    transcriptions_written = 0
    skipped = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Обработка"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('|', 1)
            if len(parts) < 2:
                continue
            
            stem, text = parts[0].strip(), parts[1].strip()
            wav_src = Path(src_dir) / f'{stem}.wav'
            
            if not wav_src.is_file():
                skipped += 1
                continue
            
            # Копирование wav
            dest_wav = Path(dst_audio_dir) / f'{stem}.wav'
            shutil.copy2(wav_src, dest_wav)
            wavs_copied += 1
            
            # Запись текста
            txt_path = Path(dst_text_dir) / f'{stem}.txt'
            txt_path.write_text(text, encoding='utf-8')
            transcriptions_written += 1
            
            if limit and wavs_copied >= limit:
                print(f"\n Остановлено на {wavs_copied} файлах")
                break
    
    return {
        'wavs_copied': wavs_copied,
        'transcriptions': transcriptions_written,
        'skipped': skipped
    }


def preprocess_audio(wav_path, target_sr=22050, normalize=True):
    """Предобработка одного аудио файла"""
    
    audio, sr = sf.read(wav_path)
    audio = audio.astype(np.float32)
    
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # стерео -> моно
    
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    if normalize:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    
    return audio


def split_dataset(index_path, train_ratio=0.9, seed=42):
    """Разделение датасета на train/val"""
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    np.random.seed(seed)
    indices = np.random.permutation(len(index))
    split_idx = int(train_ratio * len(indices))
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_index = [index[i] for i in train_indices]
    val_index = [index[i] for i in val_indices]
    
    # Сохранение разделенных индексов
    train_path = Path(index_path).parent / 'train_index.json'
    val_path = Path(index_path).parent / 'val_index.json'
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_index, f, ensure_ascii=False, indent=2)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_index, f, ensure_ascii=False, indent=2)
    
    print(f" Train: {len(train_index)} samples")
    print(f" Val: {len(val_index)} samples")
    
    return train_index, val_index


def compute_mel_stats(dataset, mel_extractor, num_samples=100):
    """Вычисление статистик мел-спектрограмм для нормализации"""
    
    mel_sum = 0
    mel_sum_sq = 0
    count = 0
    
    for i in range(min(num_samples, len(dataset))):
        batch = dataset[i]
        audio = batch['audio'].unsqueeze(0)  # [1, T]
        
        with torch.no_grad():
            mel = mel_extractor(audio)  # [1, n_mels, T']
        
        mel_sum += mel.mean(dim=-1).sum(dim=0).cpu().numpy()
        mel_sum_sq += (mel.mean(dim=-1) ** 2).sum(dim=0).cpu().numpy()
        count += mel.shape[-1]
    
    mean = mel_sum / count
    std = np.sqrt(mel_sum_sq / count - mean ** 2)
    
    return {'mean': mean.tolist(), 'std': std.tolist()}