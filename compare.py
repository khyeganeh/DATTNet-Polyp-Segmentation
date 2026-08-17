import os

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import KvasirDataset
from models.dattnet import DATTNet


# -----------------------
# Device
# -----------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# -----------------------
# Dataset
# -----------------------
dataset = KvasirDataset(
    root_dir="datasets/Kvasir-SEG/val",
    image_size=224,
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)

print("Validation Images:", len(dataset))


# -----------------------
# Load Model
# -----------------------
model = DATTNet(num_classes=1)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
    )
)

model.to(device)
model.eval()

print("Model Loaded.")


# -----------------------
# Output Folder
# -----------------------
os.makedirs("results", exist_ok=True)


# -----------------------
# Compare Images
# -----------------------
with torch.no_grad():

    for idx, (images, masks) in enumerate(tqdm(loader)):

        if idx == 20:
            break

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        if outputs.shape[-1] != masks.shape[-1]:

            outputs = F.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        outputs = torch.sigmoid(outputs)
        outputs = (outputs > 0.5).float()

        # -----------------------
        # Convert to numpy
        # -----------------------

        image = images[0].cpu().permute(1, 2, 0).numpy()

        mask = masks[0, 0].cpu().numpy()

        pred = outputs[0, 0].cpu().numpy()

        # -----------------------
        # Plot
        # -----------------------

        fig = plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.imshow(image)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1,3,2)
        plt.imshow(mask, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1,3,3)
        plt.imshow(pred, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")

        plt.tight_layout()

        plt.savefig(
            f"results/comparison_{idx+1}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

print("\n===================================")
print("20 Comparison Images Saved!")
print("Folder : results/")
print("===================================")