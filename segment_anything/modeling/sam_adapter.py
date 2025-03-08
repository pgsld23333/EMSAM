import torch
import torch.nn as nn
from .common import LayerNorm2d

class Adapter(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Adapter, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        # self.ln1 = LayerNorm2d(hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False)
        # self.ln2 = LayerNorm2d(in_channels)
    def forward(self, x):
        x = self.conv1(x)
        # x = self.ln1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.ln2(x)
        return x