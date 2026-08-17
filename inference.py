import os

import cv2
import torch
import numpy as np
from torchvision import transforms

from models.dattnet import DATTNet

# -----------------------
# Device
# -----------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

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
# Image Path
# -----------------------
image_path = "datasets/Kvasir-SEG/images/cju0qkwl35piu0993l0dewei2.jpg"

# -----------------------
# Read Image
# -----------------------
image = cv2.imread(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

original = image.copy()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

image = transform(image)

image = image.unsqueeze(0).to(device)

# -----------------------
# Prediction
# -----------------------
with torch.no_grad():

    pred = model(image)

    pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()

pred = (pred > 0.5).astype(np.uint8)

pred = pred * 255

# -----------------------
# Save
# -----------------------
os.makedirs("outputs", exist_ok=True)

cv2.imwrite(
    "outputs/prediction.png",
    pred
)

print("Prediction Saved!")