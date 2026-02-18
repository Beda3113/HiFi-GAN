#!/usr/bin/env python
import os
import sys
import argparse
import json
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils import AttrDict, build_env, scan_checkpoint, load_checkpoint
from src.datasets.ruslan_dataset import RuslanDataset, get_dataset_filelist
from src.training.trainer import HiFiGANTrainer


def train(rank, args, h):
    """Training function for each process"""
    if h.num_gpus > 1:
        torch.distributed.init_process_group(
            backend=h.dist_config['dist_backend'],
            init_method=h.dist_config['dist_url'],
            world_size=h.dist_config['world_size'] * h.num_gpus,
            rank=rank
        )
    
    # Create trainer
    trainer = HiFiGANTrainer(h, rank)
    
    # Load checkpoint if exists
    if os.path.isdir(args.checkpoint_path):
        cp_g = scan_checkpoint(args.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(args.checkpoint_path, 'do_')
        
        if cp_g is not None and cp_do is not None:
            trainer.load_checkpoint(cp_do)
            print(f"Loaded checkpoint from {cp_do}")
    
    # Setup data
    training_files, validation_files = get_dataset_filelist(
        args.input_training_file,
        args.input_validation_file,
        args.input_wavs_dir
    )
    
    trainset = RuslanDataset(
        training_files,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        n_cache_reuse=0,
        shuffle=False if h.num_gpus > 1 else True,
        fmax_loss=h.fmax_for_loss,
        device=trainer.device,
        fine_tuning=args.fine_tuning,
        base_mels_path=args.input_mels_dir
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset) if h.num_gpus > 1 else None
    
    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True
    )
    
    if rank == 0:
        validset = RuslanDataset(
            validation_files,
            h.segment_size,
            h.n_fft,
            h.num_mels,
            h.hop_size,
            h.win_size,
            h.sampling_rate,
            h.fmin,
            h.fmax,
            False,
            False,
            n_cache_reuse=0,
            fmax_loss=h.fmax_for_loss,
            device=trainer.device,
            fine_tuning=args.fine_tuning,
            base_mels_path=args.input_mels_dir
        )
        
        val_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            batch_size=1,
            pin_memory=True,
            drop_last=True
        )
        
        writer = SummaryWriter(os.path.join(args.checkpoint_path, 'logs'))
    else:
        val_loader = None
        writer = None
    
    # Training loop
    for epoch in range(trainer.epoch, args.training_epochs):
        if rank == 0:
            print(f"Epoch: {epoch + 1}")
        
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        
        for i, batch in enumerate(train_loader):
            # Train step
            losses = trainer.train_step(batch)
            trainer.steps += 1
            
            # Logging
            if rank == 0 and trainer.steps % args.stdout_interval == 0:
                print(f"Steps: {trainer.steps}, "
                      f"Gen Loss: {losses['loss_gen']:.4f}, "
                      f"Disc Loss: {losses['loss_disc']:.4f}, "
                      f"Mel Loss: {losses['loss_mel']:.4f}")
                
                writer.add_scalar("training/gen_loss", losses['loss_gen'], trainer.steps)
                writer.add_scalar("training/disc_loss", losses['loss_disc'], trainer.steps)
                writer.add_scalar("training/mel_loss", losses['loss_mel'], trainer.steps)
                
                if args.use_wandb:
                    import wandb
                    wandb.log(losses, step=trainer.steps)
            
            # Validation
            if rank == 0 and trainer.steps % args.validation_interval == 0:
                val_err = trainer.validate(val_loader, writer)
                print(f"Validation Mel Error: {val_err:.4f}")
            
            # Checkpointing
            if rank == 0 and trainer.steps % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(args.checkpoint_path, f'g_{trainer.steps:08d}')
                trainer.save_checkpoint(checkpoint_path)
        
        # Update schedulers
        trainer.scheduler_g.step()
        trainer.scheduler_d.step()
        trainer.epoch += 1
    
    if rank == 0:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='data/LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='data/ft_dataset')
    parser.add_argument('--input_training_file', default='data/LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='data/LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='checkpoints')
    parser.add_argument('--config', default='src/configs/config.yaml')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        if args.config.endswith('.yaml'):
            import yaml
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    h = AttrDict(data)
    h.training_epochs = args.training_epochs
    
    # Save config
    build_env(args.config, 'config.json', args.checkpoint_path)
    
    # Setup
    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print(f'Batch size per GPU: {h.batch_size}')
    else:
        h.num_gpus = 0
    
    # Train
    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(args, h))
    else:
        train(0, args, h)


if __name__ == '__main__':
    main()