import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm


class ResBlock1(nn.Module):
    """
    Residual Block with dilated convolutions (Type 1)
    Used in HiFi-GAN V1 and V2
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 1)):
        super(ResBlock1, self).__init__()
        
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=self.get_padding(kernel_size, dilation[0])
            )),
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=self.get_padding(kernel_size, dilation[1])
            ))
        ])
        
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=self.get_padding(kernel_size, dilation[0])
            )),
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=self.get_padding(kernel_size, dilation[1])
            ))
        ])
        
        self.activation = nn.LeakyReLU(0.1)
    
    @staticmethod
    def get_padding(kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            xt = self.activation(x)
            xt = conv1(xt)
            xt = self.activation(xt)
            xt = conv2(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            remove_weight_norm(conv1)
            remove_weight_norm(conv2)


class ResBlock2(nn.Module):
    """
    Residual Block with dilated convolutions (Type 2)
    Used in HiFi-GAN V3
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 1)):
        super(ResBlock2, self).__init__()
        
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[0],
                padding=self.get_padding(kernel_size, dilation[0])
            )),
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1, dilation=dilation[1],
                padding=self.get_padding(kernel_size, dilation[1])
            ))
        ])
        
        self.activation = nn.LeakyReLU(0.1)
    
    @staticmethod
    def get_padding(kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)
    
    def forward(self, x):
        for conv in self.convs:
            xt = self.activation(x)
            xt = conv(xt)
            x = xt + x
        return x
    
    def remove_weight_norm(self):
        for conv in self.convs:
            remove_weight_norm(conv)


class Generator(nn.Module):
    """HiFi-GAN Generator"""
    def __init__(self, config):
        super(Generator, self).__init__()
        
        self.num_kernels = len(config["resblock_kernel_sizes"])
        self.num_upsamples = len(config["upsample_rates"])
        
        # Initial convolution
        self.conv_pre = weight_norm(nn.Conv1d(
            80, config["upsample_initial_channel"], 7, 1, padding=3
        ))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(
            config["upsample_rates"], config["upsample_kernel_sizes"]
        )):
            self.ups.append(weight_norm(nn.ConvTranspose1d(
                config["upsample_initial_channel"] // (2 ** i),
                config["upsample_initial_channel"] // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2
            )))
        
        # Residual blocks (MRF)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(
                config["resblock_kernel_sizes"],
                config["resblock_dilation_sizes"]
            )):
                if config["resblock"] == "1":
                    self.resblocks.append(ResBlock1(ch, k, d))
                else:
                    self.resblocks.append(ResBlock2(ch, k, d))
        
        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.activation = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        """
        Args:
            x: mel-spectrogram [B, 80, T]
        Returns:
            audio: [B, 1, T * hop_length * upsample_factor]
        """
        x = self.conv_pre(x)
        
        for i in range(self.num_upsamples):
            x = self.activation(x)
            x = self.ups[i](x)
            
            # Apply residual blocks (MRF)
            xs = None
            for j in range(self.num_kernels):
                res_block = self.resblocks[i * self.num_kernels + j]
                if xs is None:
                    xs = res_block(x)
                else:
                    xs += res_block(x)
            x = xs / self.num_kernels
        
        x = self.activation(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference"""
        print("Removing weight norm...")
        remove_weight_norm(self.conv_pre)
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_post)