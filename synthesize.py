#!/usr/bin/env python
import os
import sys
import argparse
import json
import torch
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.utils import AttrDict
from src.utils.audio_utils import save_wav
from src.utils.mel_processing import MelSpectrogramExtractor
from src.datasets.custom_dir_dataset import CustomDirDataset
from src.inference.inference import HiFiGANInference
from torch.utils.data import DataLoader


def load_external_tts_model(model_name):
    """Load external TTS model (e.g., from NeMo or HuggingFace)"""
    try:
        from nemo.collections.tts.models import FastPitchModel
        model = FastPitchModel.from_pretrained(model_name)
        return model
    except:
        print("Warning: NeMo not available, using dummy model")
        return None


def synthesize(args):
    """Main synthesis function"""
    print("Initializing inference...")
    
    # Load config
    with open(args.config, 'r') as f:
        if args.config.endswith('.yaml'):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    config = AttrDict(data)
    
    # Create mel extractor
    mel_extractor = MelSpectrogramExtractor(config["mel"])
    
    # Create dataset
    dataset = CustomDirDataset(
        args.input_dir,
        mel_extractor,
        mode=args.mode
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Load vocoder
    vocoder = HiFiGANInference(
        args.checkpoint_file,
        config["model"]["generator"],
        args.device
    )
    
    # Load external TTS model for end-to-end mode
    tts_model = None
    if args.mode == "end_to_end" and args.tts_model:
        print(f"Loading TTS model: {args.tts_model}")
        tts_model = load_external_tts_model(args.tts_model)
        if tts_model:
            tts_model.eval().to(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each sample
    for batch in dataloader:
        file_id = batch["file_id"][0]
        
        if args.mode == "resynthesize":
            # Use ground truth audio
            mel = batch["mel"][0].numpy()
            audio = vocoder.mel_to_wav(mel)
            
            output_path = os.path.join(args.output_dir, f"{file_id}_resyn.wav")
            save_wav(audio, output_path, config["mel"]["sampling_rate"])
            print(f"Saved: {output_path}")
            
        else:  # end_to_end mode
            text = batch["text"][0]
            
            if tts_model is None:
                print(f"Warning: No TTS model available for {file_id}")
                continue
            
            # Generate mel from text
            with torch.no_grad():
                parsed = tts_model.parse(text)
                mel = tts_model.generate_spectrogram(tokens=parsed)
                mel = mel.squeeze(0).cpu().numpy()
            
            # Generate audio from mel
            audio = vocoder.mel_to_wav(mel)
            
            output_path = os.path.join(args.output_dir, f"{file_id}_tts.wav")
            save_wav(audio, output_path, config["mel"]["sampling_rate"])
            print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                       help='Path to dataset directory (with audio/ and transcriptions/)')
    parser.add_argument('--output_dir', default='generated_files',
                       help='Directory to save generated audio')
    parser.add_argument('--checkpoint_file', required=True,
                       help='Path to generator checkpoint')
    parser.add_argument('--config', default='src/configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', choices=['resynthesize', 'end_to_end'],
                       default='resynthesize',
                       help='Inference mode')
    parser.add_argument('--tts_model', default=None,
                       help='TTS model name for end-to-end mode (e.g., "bene-ges/tts_ru_ipa_fastpitch_ruslan")')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    synthesize(args)


if __name__ == '__main__':
    main()