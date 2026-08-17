import os
import random
import shutil

random.seed(42)

ROOT = "datasets/Kvasir-SEG"

image_dir = os.path.join(ROOT, "images")
mask_dir = os.path.join(ROOT, "masks")

train_img = os.path.join(ROOT, "train", "images")
train_mask = os.path.join(ROOT, "train", "masks")

val_img = os.path.join(ROOT, "val", "images")
val_mask = os.path.join(ROOT, "val", "masks")

for folder in [train_img, train_mask, val_img, val_mask]:
    os.makedirs(folder, exist_ok=True)

images = sorted(os.listdir(image_dir))

random.shuffle(images)

split = int(len(images) * 0.8)

train_images = images[:split]
val_images = images[split:]

print("Train:", len(train_images))
print("Validation:", len(val_images))

for name in train_images:
    shutil.copy(
        os.path.join(image_dir, name),
        os.path.join(train_img, name)
    )
    shutil.copy(
        os.path.join(mask_dir, name),
        os.path.join(train_mask, name)
    )

for name in val_images:
    shutil.copy(
        os.path.join(image_dir, name),
        os.path.join(val_img, name)
    )
    shutil.copy(
        os.path.join(mask_dir, name),
        os.path.join(val_mask, name)
    )

print("Done!")