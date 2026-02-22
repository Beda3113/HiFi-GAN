"""Пакет с конфигурациями"""

from .model_config import mel_config, generator_config
from .training_config import training_config, data_config

__all__ = [
    'mel_config',
    'generator_config',
    'training_config',
    'data_config'
]