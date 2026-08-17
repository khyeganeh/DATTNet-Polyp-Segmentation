import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class VGGEncoder(nn.Module):
    """
    VGG16-BN Encoder for DATTNet
    Returns feature maps from 5 stages.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            backbone = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
        else:
            backbone = vgg16_bn(weights=None)

        features = backbone.features

        self.stage1 = features[:6]      # 64
        self.stage2 = features[6:13]    # 128
        self.stage3 = features[13:23]   # 256
        self.stage4 = features[23:33]   # 512
        self.stage5 = features[33:43]   # 512

    def forward(self, x):

        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)

        return f1, f2, f3, f4, f5