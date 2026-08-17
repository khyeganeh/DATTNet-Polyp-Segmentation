import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KvasirDataset
from models.dattnet import DATTNet
from metrics import SegmentationMetrics

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
    train=False,
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
)

print("Dataset Size:", len(dataset))

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
# Metrics
# -----------------------
metrics = SegmentationMetrics()

dice_total = 0.0
iou_total = 0.0
precision_total = 0.0
recall_total = 0.0
f1_total = 0.0
accuracy_total = 0.0

print("\nStarting Evaluation...\n")

with torch.no_grad():

    for images, masks in tqdm(loader, desc="Evaluating"):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # اگر خروجی مدل 448x448 بود، به اندازه ماسک تبدیل کن
        if outputs.shape[-1] != masks.shape[-1]:

            outputs = F.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        result = metrics(outputs, masks)

        dice_total += result["dice"]
        iou_total += result["iou"]
        precision_total += result["precision"]
        recall_total += result["recall"]
        f1_total += result["f1"]
        accuracy_total += result["accuracy"]

# -----------------------
# Average Metrics
# -----------------------
num_samples = len(loader)

dice = dice_total / num_samples
iou = iou_total / num_samples
precision = precision_total / num_samples
recall = recall_total / num_samples
f1 = f1_total / num_samples
accuracy = accuracy_total / num_samples

print("\n==============================")
print("Evaluation Results")
print("==============================")
print(f"Dice Score : {dice:.4f}")
print(f"IoU Score  : {iou:.4f}")
print(f"Precision  : {precision:.4f}")
print(f"Recall     : {recall:.4f}")
print(f"F1 Score   : {f1:.4f}")
print(f"Accuracy   : {accuracy:.4f}")
print("==============================")