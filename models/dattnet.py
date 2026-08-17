import torch
import torch.nn as nn

from models.vgg_encoder import VGGEncoder
from models.bridge import Bridge
from models.decoder import DecoderBlock
from models.dual_attention import DualAttention


class DATTNet(nn.Module):

    def __init__(self, num_classes=1):
        super().__init__()

        # ---------------- Encoder ----------------
        self.encoder = VGGEncoder(pretrained=True)

        # ---------------- Bridge ----------------
        self.bridge = Bridge()

        # ---------------- Dual Attention ----------------
        self.da5 = DualAttention(512)
        self.da4 = DualAttention(512)
        self.da3 = DualAttention(256)
        self.da2 = DualAttention(128)
        self.da1 = DualAttention(64)

        # ---------------- Decoder ----------------
        self.decoder4 = DecoderBlock(512, 512, 256)
        self.decoder3 = DecoderBlock(256, 256, 128)
        self.decoder2 = DecoderBlock(128, 128, 64)
        self.decoder1 = DecoderBlock(64, 64, 64)

        # ---------------- Segmentation Head ----------------
        self.seg_head = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x):

        # Encoder
        f1, f2, f3, f4, f5 = self.encoder(x)

        # Attention
        f1 = self.da1(f1)
        f2 = self.da2(f2)
        f3 = self.da3(f3)
        f4 = self.da4(f4)
        f5 = self.da5(f5)

        # Bridge
        x = self.bridge([f1, f2, f3, f4, f5])

        # Decoder
        x = self.decoder4(x, f4)
        x = self.decoder3(x, f3)
        x = self.decoder2(x, f2)
        x = self.decoder1(x, f1)

        # Segmentation
        x = self.seg_head(x)

        return x