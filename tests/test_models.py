import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.generator import Generator
from src.models.discriminator import MultiPeriodDiscriminator, MultiScaleDiscriminator


def test_generator():
    """Test generator forward pass"""
    config = {
        "resblock": "1",
        "upsample_rates": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    }
    
    generator = Generator(config)
    mel = torch.randn(2, 80, 32)  # [B, C, T]
    audio = generator(mel)
    
    assert audio.shape[0] == 2
    assert audio.shape[1] == 1
    assert audio.shape[2] == 32 * 256  # 256 = product of upsample_rates
    
    print("Generator test passed!")


def test_discriminators():
    """Test discriminators forward pass"""
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()
    
    y = torch.randn(2, 1, 8192)
    y_hat = torch.randn(2, 1, 8192)
    
    # Test MPD
    mpd_y_d_rs, mpd_y_d_gs, fmap_rs, fmap_gs = mpd(y, y_hat)
    assert len(mpd_y_d_rs) == 5
    assert len(mpd_y_d_gs) == 5
    
    # Test MSD
    msd_y_d_rs, msd_y_d_gs, fmap_rs, fmap_gs = msd(y, y_hat)
    assert len(msd_y_d_rs) == 3
    
    print("Discriminators test passed!")


if __name__ == "__main__":
    test_generator()
    test_discriminators()