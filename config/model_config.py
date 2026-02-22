"""Конфигурация модели HiFi-GAN"""

mel_config = {
    'sample_rate': 22050,
    'n_mels': 80,
    'n_fft': 1024,
    'hop_length': 256,
    'win_length': 1024,
    'f_min': 0,
    'f_max': 8000,
}

generator_config = {
    'n_mels': 80,
    'hidden_dim': 512,
    'upsample_rates': [8, 8, 2, 2],
    'upsample_kernel_sizes': [16, 16, 4, 4],
    'kernel_sizes': [3, 7, 11],
    'dilation_rates': [1, 3, 5],
}