

# HiFi-GAN для RUSLAN датасета

Реализация HiFi-GAN для синтеза речи на русском языке с использованием датасета RUSLAN.

## Особенности
- Полная реализация HiFi-GAN (Generator + MPD + MSD)
- Адаптировано для русского языка
- Поддержка аугментации данных
- Gradient accumulation для эффективного обучения
- TensorBoard визуализация

## Требования
- Python 3.8+
- PyTorch 2.0+
- CUDA (опционально)

## Установка
```bash
git clone https://github.com/yourusername/hifigan-ruslan
cd hifigan-ruslan
pip install -r requirements.txt


Подготовка данных

python scripts/prepare_data.py --data_path /path/to/ruslan


Обучение

python scripts/train.py --config config/training_config.py


Генерация

python scripts/generate_samples.py --checkpoint checkpoints/generator_best.pt


Структура
config/ - конфигурационные файлы

data/ - датасет и preprocessing

models/ - архитектуры моделей

training/ - логика обучения

scripts/ - исполняемые скрипты





### `requirements.txt`
```txt
torch
torchaudio
librosa
soundfile
numpy
pandas
tqdm
tensorboard
matplotlib
scipy