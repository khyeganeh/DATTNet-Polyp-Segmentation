import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (CBAM Style)
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7)

        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg = torch.mean(x, dim=1, keepdim=True)

        mx, _ = torch.max(x, dim=1, keepdim=True)

        att = torch.cat([avg, mx], dim=1)

        att = self.conv(att)

        att = self.sigmoid(att)

        return x * att