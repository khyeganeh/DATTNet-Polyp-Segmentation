import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class VGGEncoder(nn.Module):
    """
    VGG16-BN Encoder

    Outputs:
        x1 : [B, 64, 224,224]
        x2 : [B,128,112,112]
        x3 : [B,256, 56, 56]
        x4 : [B,512, 28, 28]
        x5 : [B,512, 14, 14]
    """

    def __init__(self, pretrained=True):
        super().__init__()

        if pretrained:
            weights = VGG16_BN_Weights.DEFAULT
        else:
            weights = None

        features = vgg16_bn(weights=weights).features

        self.stage1 = nn.Sequential(*features[:6])
        self.stage2 = nn.Sequential(*features[6:13])
        self.stage3 = nn.Sequential(*features[13:23])
        self.stage4 = nn.Sequential(*features[23:33])
        self.stage5 = nn.Sequential(*features[33:43])

    def forward(self, x):

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x1, x2, x3, x4, x5