import torch
import torch.nn as nn
import torch.nn.functional as F


class Bridge(nn.Module):
    """
    Multi-scale Feature Bridge
    """

    def __init__(self, channels=(64, 128, 256, 512, 512), out_channels=512):
        super().__init__()

        self.proj = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1, bias=False)
            for c in channels
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features):
        """
        features:
            [f1,f2,f3,f4,f5]
        """

        target_size = features[-1].shape[2:]

        outs = []

        for feat, proj in zip(features, self.proj):

            x = proj(feat)

            if x.shape[2:] != target_size:
                x = F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            outs.append(x)

        x = torch.cat(outs, dim=1)

        x = self.fuse(x)

        return x