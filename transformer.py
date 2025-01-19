import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm='instance'):
        super().__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        
        self.norm = norm
        if norm == 'instance':
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        
        # Difference between instance and batch normalization
        # Instance Normalization is used for style transfer because it normalizes the activations of the features in the image
        # Batch Normalization is used for training the network because it normalizes the activations of the features in the network
    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv(out)
        if self.norm == 'None':
            return out
        out = self.norm_layer(out)
        return out
    
class ResidualBlock(nn.Module):
    def __init__(self, channels=128, kernel_size=3):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size, stride=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x
    
class DeconvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding=1, norm='instance'):
        super().__init__()
        padding = kernel_size // 2
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        
        self.norm = norm
        if norm == 'instance':
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm == 'batch':
            self.norm_layer = nn.BatchNorm2d(out_channels, affine=True)
        
    def forward(self, x):
        out = self.deconv(x)
        if self.norm == 'None':
            return out
        out = self.norm_layer(out)
        return out

class TransformerNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ConvBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.ReLU(),
            ConvLayer(32, 64, 3, 2),
            nn.ReLU(),
            ConvLayer(64, 128, 3, 2),
            nn.ReLU()
        )
        
        self.ResidualBlock = nn.Sequential(
            ResidualBlock(128,3),
            ResidualBlock(128,3),
            ResidualBlock(128,3),
            ResidualBlock(128,3),
            ResidualBlock(128,3)
        )
        
        self.DeconvBlock = nn.Sequential(
            DeconvLayer(128, 64, 3, 2, 1),
            nn.ReLU(),
            DeconvLayer(64, 32, 3, 2, 1),
            nn.ReLU(),
            ConvLayer(32, 3, 9, 1, norm='None')
        )
        
    def forward(self, x):
        out = self.ConvBlock(x)
        out = self.ResidualBlock(out)
        out = self.DeconvBlock(out)
        return out
    
class TransformerNetworkTanh(TransformerNetwork):
    def __init__(self, tanh_multiplier=150, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tanh = nn.Tanh()
        self.tanh_multiplier = tanh_multiplier
        
    def forward(self, x):
        out = super().forward(x)
        out = self.tanh(out) * self.tanh_multiplier
        return out
