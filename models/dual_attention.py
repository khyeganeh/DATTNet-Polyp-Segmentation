import torch
import torch.nn as nn

from models.attention import ChannelAttention
from models.spatial_attention import SpatialAttention
from models.eca import ECABlock


class DualAttention(nn.Module):
    """
    Dual Attention Module
    Channel Attention + Spatial Attention + ECA
    """

    def __init__(self, channels):
        super().__init__()

        self.channel = ChannelAttention(channels)
        self.spatial = SpatialAttention()
        self.eca = ECABlock(channels)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        c = self.channel(x)

        s = self.spatial(x)

        e = self.eca(x)

        out = torch.cat([c, s, e], dim=1)

        out = self.fuse(out)

        return out + x