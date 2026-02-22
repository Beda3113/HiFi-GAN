#!/usr/bin/env python
"""Скрипт для подготовки данных RUSLAN"""

import argparse
import json
import shutil
from pathlib import Path
from tqdm import tqdm


def prepare_data(src_dir, dst_dir, limit=None):
    """Подготовка данных для обучения"""
    
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    audio_dir = dst_dir / 'audio'
    text_dir = dst_dir / 'transcriptions'
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Поиск файлов в {src_dir}...")
    
    # Поиск wav файлов
    wav_files = list(src_dir.glob('**/*.wav'))
    print(f"Найдено wav файлов: {len(wav_files)}")
    
    if limit:
        wav_files = wav_files[:limit]
        print(f"Ограничено: {limit} файлов")
    
    index = []
    
    print("Копирование файлов...")
    for wav_path in tqdm(wav_files):
        stem = wav_path.stem
        dest_wav = audio_dir / f'{stem}.wav'
        
        # Копирование wav
        shutil.copy2(wav_path, dest_wav)
        
        # Поиск соответствующего txt
        txt_candidates = list(src_dir.glob(f'**/{stem}.txt'))
        if txt_candidates:
            text = txt_candidates[0].read_text(encoding='utf-8').strip()
        else:
            text = "Текст транскрипции"
        
        # Сохранение текста
        txt_path = text_dir / f'{stem}.txt'
        txt_path.write_text(text, encoding='utf-8')
        
        # Добавление в индекс
        index.append({
            'path': str(dest_wav.resolve()),
            'text': text,
            'has_audio': True
        })
    
    # Сохранение индекса
    index_path = dst_dir / 'custom_dir_index.json'
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    
    print(f"\nГотово!")
    print(f"Скопировано: {len(index)} файлов")
    print(f"Индекс сохранен: {index_path}")
    
    return index


def main():
    parser = argparse.ArgumentParser(description='Подготовка RUSLAN датасета')
    parser.add_argument('--src', type=str, required=True,
                       help='Путь к исходным данным')
    parser.add_argument('--dst', type=str, default='/kaggle/working/data/ruslan',
                       help='Путь для сохранения')
    parser.add_argument('--limit', type=int, default=None,
                       help='Ограничить количество файлов')
    
    args = parser.parse_args()
    
    prepare_data(args.src, args.dst, args.limit)


if __name__ == '__main__':
    main()