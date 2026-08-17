import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset


class KvasirDataset(Dataset):

    def __init__(self, root_dir, image_size=224, train=True):
        super().__init__()

        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")

        self.image_size = image_size
        self.train = train

        self.images = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

        # -------------------------
        # Train Transform
        # -------------------------
        self.train_transform = A.Compose([
            A.Resize(image_size, image_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.Rotate(
                limit=20,
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),

            A.GaussianBlur(
                blur_limit=(3, 5),
                p=0.2,
            ),

            A.Normalize(),

            ToTensorV2(),
        ])

        # -------------------------
        # Validation/Test Transform
        # -------------------------
        self.val_transform = A.Compose([
            A.Resize(image_size, image_size),

            A.Normalize(),

            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_name = self.images[index]

        image_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Read Image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Binary Mask
        mask = (mask > 127).astype(np.float32)

        # Apply Transform
        if self.train:
            transformed = self.train_transform(
                image=image,
                mask=mask,
            )
        else:
            transformed = self.val_transform(
                image=image,
                mask=mask,
            )

        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0).float()

        return image, mask