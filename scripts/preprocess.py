#!/usr/bin/env python
import os
import argparse
import torch
import librosa
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.audio_utils import load_wav, normalize_audio
from src.utils.mel_processing import MelSpectrogramExtractor


def preprocess(args):
    """Preprocess dataset: resample and extract mel-spectrograms"""
    print(f"Preprocessing dataset from {args.input_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "mels"), exist_ok=True)
    
    # Find all wav files
    wav_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(wav_files)} audio files")
    
    # Create mel extractor
    mel_config = {
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
        "sampling_rate": args.sampling_rate,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
        "f_min": args.f_min,
        "f_max": args.f_max,
    }
    mel_extractor = MelSpectrogramExtractor(mel_config)
    
    # Process each file
    for wav_path in tqdm(wav_files):
        # Load and resample
        audio, sr = load_wav(wav_path, args.sampling_rate)
        audio = normalize_audio(audio)
        audio = torch.FloatTensor(audio)
        
        # Extract mel
        mel = mel_extractor(audio.unsqueeze(0)).squeeze(0).numpy()
        
        # Save mel
        rel_path = os.path.relpath(wav_path, args.input_dir)
        mel_name = os.path.splitext(rel_path)[0].replace('/', '_') + '.npy'
        mel_path = os.path.join(args.output_dir, "mels", mel_name)
        np.save(mel_path, mel)
    
    print("Preprocessing complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Input dataset directory')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--sampling_rate', type=int, default=22050, help='Target sampling rate')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    parser.add_argument('--n_mels', type=int, default=80, help='Number of mel bands')
    parser.add_argument('--hop_length', type=int, default=256, help='Hop length')
    parser.add_argument('--win_length', type=int, default=1024, help='Window length')
    parser.add_argument('--f_min', type=int, default=0, help='Minimum frequency')
    parser.add_argument('--f_max', type=int, default=8000, help='Maximum frequency')
    
    args = parser.parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()