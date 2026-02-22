"""Пакет для работы с данными RUSLAN"""

from .dataset import RUSLANDataset
from .preprocessing import (
    create_index,
    extract_metadata_from_csv,
    preprocess_audio,
    split_dataset,
    compute_mel_stats
)

__all__ = [
    'RUSLANDataset',
    'create_index',
    'extract_metadata_from_csv',
    'preprocess_audio',
    'split_dataset',
    'compute_mel_stats'
]