import torch
import torch.nn as nn
from functools import partial

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=None):
        super(ConvBlock, self).__init__()
        if kernel_size:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=kernel_size)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
    
    def forward(self, x):
        return self.block(x)

"""
Model used in Feature Extractor
"""
class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()

        # add padding so dimensions are preserved
        self.layers = nn.Sequential(
            ConvBlock(1, 64, kernel_size=2),
            ConvBlock(64, 64, kernel_size=2),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.Flatten(),
            nn.Linear(3136, 1623)
        )

        self.activations = dict() # store activations from each layer
        self.add_forward_hooks()

    def forward(self, x):
        return self.layers(x)

    def add_forward_hooks(self):
        def activation(name, module, inp, out):
            self.activations[name] = out
        
        for name, module in self.named_modules():
            module.register_forward_hook(partial(activation, name))

"""
Model used for Generator
"""
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=128))

    def forward(self, x):
        return self.fc(x)
