import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextFusionBridge(nn.Module):

    def __init__(self, ch4=512, ch5=512):
        super().__init__()

        self.conv4 = nn.Sequential(
            nn.Conv2d(ch4, ch4, 3, padding=1),
            nn.BatchNorm2d(ch4),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(ch5, ch5, 3, padding=1),
            nn.BatchNorm2d(ch5),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(ch4 + ch5, ch4, 3, padding=1),
            nn.BatchNorm2d(ch4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x4, x5):

        x5 = F.interpolate(
            x5,
            size=x4.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        x = torch.cat([x4, x5], dim=1)

        x = self.fusion(x)

        return x