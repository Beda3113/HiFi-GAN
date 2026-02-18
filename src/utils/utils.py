import os
import glob
import torch
import shutil
import matplotlib.pyplot as plt
import numpy as np


class AttrDict(dict):
    """Dictionary that allows attribute access"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    """Save configuration file to checkpoint directory"""
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def plot_spectrogram(spectrogram):
    """Plot spectrogram for TensorBoard"""
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()
    
    # Convert to numpy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def scan_checkpoint(cp_dir, prefix):
    """Scan for latest checkpoint"""
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def load_checkpoint(filepath, device):
    """Load checkpoint"""
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    """Save checkpoint"""
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")