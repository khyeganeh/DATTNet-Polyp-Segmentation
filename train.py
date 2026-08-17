import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KvasirDataset
from models.dattnet import DATTNet
from loss import BCEDiceLoss
from early_stopping import EarlyStopping

# -----------------------
# Device
# -----------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

# -----------------------
# Hyperparameters
# -----------------------
IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

# -----------------------
# Train Dataset
# -----------------------
train_dataset = KvasirDataset(
    root_dir="datasets/Kvasir-SEG/train",
    image_size=IMAGE_SIZE,
    train=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# -----------------------
# Validation Dataset
# -----------------------
val_dataset = KvasirDataset(
    root_dir="datasets/Kvasir-SEG/val",
    image_size=IMAGE_SIZE,
    train=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

print("Train Images :", len(train_dataset))
print("Validation   :", len(val_dataset))

# -----------------------
# Model
# -----------------------
model = DATTNet(num_classes=1).to(device)

# -----------------------
# Loss
# -----------------------
criterion = BCEDiceLoss()

# -----------------------
# Optimizer
# -----------------------
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
)

# -----------------------
# Early Stopping
# -----------------------
early_stopping = EarlyStopping(
    patience=5,
    delta=0.001,
)

best_val_loss = float("inf")

os.makedirs("checkpoints", exist_ok=True)

# -----------------------
# History
# -----------------------
train_history = []
val_history = []

# =====================================================
# Training
# =====================================================
for epoch in range(EPOCHS):

    print(f"\n========== Epoch {epoch+1}/{EPOCHS} ==========")

    # ---------------- TRAIN ----------------
    model.train()

    train_loss = 0.0

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{EPOCHS}"
    )

    for images, masks in train_bar:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        if outputs.shape[-1] != masks.shape[-1]:
            outputs = F.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        train_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader)

    # ---------------- VALIDATION ----------------
    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

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

            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    # -----------------------
    # Save History
    # -----------------------
    train_history.append(train_loss)
    val_history.append(val_loss)

    print(f"\nTrain Loss      : {train_loss:.4f}")
    print(f"Validation Loss : {val_loss:.4f}")
    print(f"Learning Rate   : {current_lr:.6f}")

    # -----------------------
    # Early Stopping
    # -----------------------
    early_stopping(val_loss, model)

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    if early_stopping.early_stop:

        print("\n🛑 Training stopped early.")

        break

# -----------------------
# Save Final Model
# -----------------------
torch.save(
    model.state_dict(),
    "checkpoints/dattnet_final.pth",
)

# -----------------------
# Save Training History
# -----------------------
history = pd.DataFrame({
    "Epoch": range(1, len(train_history) + 1),
    "Train Loss": train_history,
    "Validation Loss": val_history,
})

history.to_csv("history.csv", index=False)

print("✅ history.csv saved.")

print("\n===================================")
print("🎉 Training Completed")
print(f"Best Validation Loss : {best_val_loss:.4f}")
print("History File         : history.csv")
print("===================================")