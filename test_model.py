# -*- coding: utf-8 -*-
"""
Программа для проверки работы обученной модели HiFi-GAN
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================================
# Конфигурация (должна совпадать с обучением)
# ============================================================================
MEL_CONFIG = {
    'sample_rate': 22050,
    'n_mels': 80,
    'n_fft': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'f_min': 0,
    'f_max': 8000,
}

GENERATOR_CONFIG = {
    'n_mels': 80,
    'hidden_dim': 512,
    'upsample_rates': [8, 8, 2, 2],
    'upsample_kernel_sizes': [16, 16, 4, 4],
    'kernel_sizes': [3, 7, 11],
    'dilation_rates': [1, 3, 5],
}


# ============================================================================
# Модель (копия архитектуры из обучения)
# ============================================================================

def _pad(k: int, d: int) -> int:
    return (k * d - d) // 2

def _init_weights(m, std=0.01):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


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
        # audio должен быть на том же устройстве, что и модель
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        
        # Убеждаемся, что окно на том же устройстве
        hann_window = self.hann_window.to(audio.device)
        
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            return_complex=True
        )
        mag = torch.abs(spec)
        mel = torch.matmul(self.mel_basis.to(audio.device), mag)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel


class MRFBlock(nn.Module):
    """Multi-Receptive Field Fusion блок"""
    def __init__(self, channels, kernel_sizes, dilation_rates):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv1d(channels, channels, k, dilation=d, padding=_pad(k, d)))
            for k in kernel_sizes for d in dilation_rates
        ])
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        for conv in self.convs:
            x = x + self.act(conv(x))
        return x


class Generator(nn.Module):
    """Генератор HiFi-GAN"""
    def __init__(self, config):
        super().__init__()
        n_mels = config['n_mels']
        hu = config['hidden_dim']
        ku = config['upsample_kernel_sizes']
        kr = config['kernel_sizes']
        Dr = config['dilation_rates']
        upsample_rates = config['upsample_rates']
        
        self.enc_conv = nn.utils.weight_norm(nn.Conv1d(n_mels, hu, 7, padding=3))
        self.act = nn.LeakyReLU(0.1)
        
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        for i in range(len(ku)):
            in_ch = hu // (2 ** i)
            out_ch = hu // (2 ** (i + 1))
            up = nn.ConvTranspose1d(in_ch, out_ch, ku[i], stride=upsample_rates[i], padding=ku[i]//2)
            self.ups.append(nn.utils.weight_norm(up))
            self.mrfs.append(MRFBlock(out_ch, kr, Dr))
        
        self.final_conv = nn.utils.weight_norm(nn.Conv1d(out_ch, 1, 7, padding='same'))
    
    def forward(self, mel):
        x = self.act(self.enc_conv(mel))
        for up, mrf in zip(self.ups, self.mrfs):
            x = mrf(up(x))
            x = self.act(x)
        fake_waveform = torch.tanh(self.final_conv(x))
        return fake_waveform


class HiFiGAN(nn.Module):
    """Полная модель HiFi-GAN (только генератор для инференса)"""
    def __init__(self, generator_config, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.generator = Generator(generator_config).to(self.device)
        self.mel_extractor = MelSpectrogram(MEL_CONFIG).to(self.device)
    
    def forward(self, mel):
        return self.generator(mel)
    
    @torch.no_grad()
    def synthesize(self, audio_path=None, audio_tensor=None, text=None):
        """
        Синтез аудио из:
        - аудиофайла (ресинтез)
        - тензора аудио
        - текста (если есть акустическая модель)
        """
        if audio_path is not None:
            # Загрузка аудио из файла
            audio, sr = sf.read(audio_path)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            if sr != MEL_CONFIG['sample_rate']:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=MEL_CONFIG['sample_rate'])
            audio = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)
        
        elif audio_tensor is not None:
            audio = audio_tensor
        
        else:
            raise ValueError("Необходимо указать audio_path или audio_tensor")
        
        # Перемещаем аудио на то же устройство, что и модель
        audio = audio.to(self.device)
        
        # Извлечение мел-спектрограммы
        mel = self.mel_extractor(audio)
        
        # Генерация аудио
        fake_audio = self.generator(mel)
        
        return {
            'original': audio.squeeze().cpu().numpy(),
            'generated': fake_audio.squeeze().cpu().numpy(),
            'mel': mel.squeeze().cpu().numpy(),
            'sr': MEL_CONFIG['sample_rate']
        }


# ============================================================================
# Утилиты для визуализации
# ============================================================================

def plot_comparison(original, generated, sr, save_path=None):
    """Построение сравнительных графиков"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # Формы волн
    time_orig = np.arange(len(original)) / sr
    time_gen = np.arange(len(generated)) / sr
    
    axes[0, 0].plot(time_orig[:min(5000, len(original))], original[:min(5000, len(original))])
    axes[0, 0].set_title('Оригинал (форма волны)')
    axes[0, 0].set_xlabel('Время (с)')
    axes[0, 0].set_ylabel('Амплитуда')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time_gen[:min(5000, len(generated))], generated[:min(5000, len(generated))])
    axes[0, 1].set_title('Генерация (форма волны)')
    axes[0, 1].set_xlabel('Время (с)')
    axes[0, 1].set_ylabel('Амплитуда')
    axes[0, 1].grid(True)
    
    # Спектрограммы
    axes[1, 0].specgram(original, Fs=sr, NFFT=1024, noverlap=512)
    axes[1, 0].set_title('Оригинал (спектрограмма)')
    axes[1, 0].set_xlabel('Время (с)')
    axes[1, 0].set_ylabel('Частота (Гц)')
    
    axes[1, 1].specgram(generated, Fs=sr, NFFT=1024, noverlap=512)
    axes[1, 1].set_title('Генерация (спектрограмма)')
    axes[1, 1].set_xlabel('Время (с)')
    axes[1, 1].set_ylabel('Частота (Гц)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"График сохранён: {save_path}")
    
    plt.show()


def analyze_frequency(audio, sr, title="Спектр частот"):
    """Анализ частотного спектра"""
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft)
    
    plt.figure(figsize=(12, 4))
    plt.semilogy(freqs[:sr//2], magnitude[:sr//2])
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда')
    plt.title(title)
    plt.grid(True)
    plt.xlim(0, 4000)  # Речевой диапазон
    plt.show()
    
    # Основная частота
    if len(magnitude[:4000]) > 0:
        main_freq = freqs[np.argmax(magnitude[:4000])]
        print(f"Основная частота: {main_freq:.1f} Гц")
        return main_freq
    return None


# ============================================================================
# Основная программа
# ============================================================================

def find_model_file(base_dir):
    """Поиск файлов модели .pt в директории"""
    pt_files = list(Path(base_dir).glob('*.pt'))
    if pt_files:
        print(f"Найдены файлы моделей:")
        for i, f in enumerate(pt_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {i+1}. {f.name} ({size_mb:.1f} MB)")
        return pt_files
    return []


def main():
    parser = argparse.ArgumentParser(description='Проверка модели HiFi-GAN')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Путь к чекпоинту модели (если не указан, будет поиск)')
    parser.add_argument('--data_dir', type=str,
                        default=r'C:\Users\user\Desktop\HiFi-GAN\data\ruslan',
                        help='Путь к папке с аудиофайлами')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Папка для сохранения результатов')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Количество тестовых файлов')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Устройство для инференса (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Создание папки для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Определение пути к модели
    base_dir = Path(__file__).parent.absolute()
    
    if args.model_path is None:
        # Поиск файлов модели
        model_files = find_model_file(base_dir)
        if not model_files:
            print("Файлы модели .pt не найдены!")
            print(f"Искали в: {base_dir}")
            return
        
        # Автоматически выбираем checkpoint_025.pt если он есть
        target_model = base_dir / 'checkpoint_025.pt'
        if target_model.exists():
            model_path = target_model
            print(f"\nВыбрана модель: {model_path.name}")
        else:
            # Иначе берем первый найденный
            model_path = model_files[0]
            print(f"\nВыбрана модель: {model_path.name}")
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Файл модели не найден: {model_path}")
            return
    
    # Устройство
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    print(f"\nЗагрузка модели из {model_path}")
    
    # Загрузка чекпоинта с обработкой разных форматов
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("Чекпоинт загружен успешно")
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print("Чекпоинт загружен со второй попытки")
        except Exception as e2:
            print(f"Не удалось загрузить чекпоинт: {e2}")
            return
    
    model = HiFiGAN(GENERATOR_CONFIG, device=device)
    
    # Извлечение весов генератора из чекпоинта
    try:
        if isinstance(checkpoint, dict):
            if 'generator' in checkpoint:
                model.generator.load_state_dict(checkpoint['generator'])
                print("Загружены веса из 'generator'")
            elif 'model_state_dict' in checkpoint:
                # Если это полный чекпоинт с обучением
                state_dict = checkpoint['model_state_dict']
                # Фильтруем только ключи генератора
                generator_state = {k.replace('generator.', ''): v 
                                  for k, v in state_dict.items() 
                                  if k.startswith('generator.')}
                if generator_state:
                    model.generator.load_state_dict(generator_state)
                    print("Загружены веса из 'model_state_dict' (отфильтрованы)")
                else:
                    model.generator.load_state_dict(state_dict)
                    print("Загружены веса из 'model_state_dict' (прямая загрузка)")
            elif 'model' in checkpoint:
                model.generator.load_state_dict(checkpoint['model'])
                print("Загружены веса из 'model'")
            else:
                # Пробуем загрузить напрямую
                try:
                    model.generator.load_state_dict(checkpoint)
                    print("Загружены веса напрямую из чекпоинта")
                except Exception as e:
                    print(f"Не удалось загрузить веса: {e}")
                    # Пробуем загрузить с адаптацией ключей
                    new_state_dict = {}
                    for k, v in checkpoint.items():
                        if k.startswith('generator.'):
                            new_key = k.replace('generator.', '')
                            new_state_dict[new_key] = v
                    if new_state_dict:
                        model.generator.load_state_dict(new_state_dict)
                        print("Загружены веса после адаптации ключей")
                    else:
                        print("Не удалось загрузить веса ни одним из способов")
                        return
        else:
            model.generator.load_state_dict(checkpoint)
            print("Загружены веса (прямая загрузка)")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {e}")
        import traceback
        traceback.print_exc()
        return
    
    model.eval()
    print("Модель загружена успешно")
    
    # Проверка наличия аудиофайлов
    if not Path(args.data_dir).exists():
        print(f"Папка с данными не найдена: {args.data_dir}")
        # Ищем альтернативные пути
        alt_paths = [
            Path(base_dir) / 'data' / 'ruslan',
            Path(base_dir) / 'ruslan',
            Path(base_dir) / 'audio',
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                args.data_dir = str(alt_path)
                print(f"Используем альтернативный путь: {args.data_dir}")
                break
    
    # Поиск аудиофайлов
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Папка с данными не найдена: {data_path}")
        return
    
    audio_files = list(data_path.glob('**/*.wav'))
    if not audio_files:
        audio_files = list(data_path.glob('**/*.WAV'))
    if not audio_files:
        audio_files = list(data_path.glob('**/*.mp3'))
    
    print(f"\nНайдено аудиофайлов: {len(audio_files)}")
    
    if len(audio_files) == 0:
        print("Аудиофайлы не найдены!")
        print("Проверьте путь к данным или укажите правильный путь через --data_dir")
        return
    
    # Выбор случайных файлов для теста
    import random
    random.seed(42)
    test_files = random.sample(audio_files, min(args.num_samples, len(audio_files)))
    
    print(f"\nТестирование на {len(test_files)} файлах")
    
    # Метрики
    mel_losses = []
    
    for i, audio_path in enumerate(tqdm(test_files, desc="Обработка")):
        try:
            # Синтез
            result = model.synthesize(audio_path=str(audio_path))
            
            # Сохранение аудио
            orig_path = os.path.join(args.output_dir, f'{i:02d}_orig.wav')
            gen_path = os.path.join(args.output_dir, f'{i:02d}_gen.wav')
            
            sf.write(orig_path, result['original'], result['sr'])
            sf.write(gen_path, result['generated'], result['sr'])
            
            # Вычисление Mel Loss
            orig_tensor = torch.FloatTensor(result['original']).unsqueeze(0).unsqueeze(0).to(device)
            gen_tensor = torch.FloatTensor(result['generated']).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                orig_mel = model.mel_extractor(orig_tensor)
                gen_mel = model.mel_extractor(gen_tensor)
                
                min_len = min(orig_mel.shape[-1], gen_mel.shape[-1])
                mel_loss = F.l1_loss(gen_mel[..., :min_len], orig_mel[..., :min_len])
                mel_losses.append(mel_loss.item())
            
            # Визуализация для первого файла
            if i == 0:
                plot_path = os.path.join(args.output_dir, f'comparison_{i:02d}.png')
                plot_comparison(result['original'], result['generated'], result['sr'], plot_path)
                analyze_frequency(result['generated'], result['sr'], "Генерация - спектр частот")
            
            print(f"\nФайл {i+1}: {audio_path.name}")
            print(f"   Mel Loss: {mel_loss:.4f}")
            
        except Exception as e:
            print(f"Ошибка при обработке {audio_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Итоговая статистика
    if mel_losses:
        print(f"\n{'='*60}")
        print("ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*60}")
        print(f"Средний Mel Loss: {np.mean(mel_losses):.4f}")
        print(f"Медианный Mel Loss: {np.median(mel_losses):.4f}")
        print(f"Стандартное отклонение: {np.std(mel_losses):.4f}")
        print(f"Минимальный: {np.min(mel_losses):.4f}")
        print(f"Максимальный: {np.max(mel_losses):.4f}")
        
        # Сравнение с результатами обучения
        expected_val_loss = 0.330  # из обучения
        print(f"\nСравнение с валидационным loss при обучении: {expected_val_loss:.4f}")
        print(f"Разница: {np.mean(mel_losses) - expected_val_loss:+.4f}")
        
        if np.mean(mel_losses) < expected_val_loss * 1.2:
            print("Качество соответствует ожидаемому")
        else:
            print("Качество ниже ожидаемого")
    
    print(f"\nРезультаты сохранены в папке: {args.output_dir}")
    print("   - .wav файлы: оригинал и генерация")
    print("   - .png файлы: сравнительные графики")


if __name__ == '__main__':
    main()