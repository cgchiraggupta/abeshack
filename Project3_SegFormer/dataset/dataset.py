import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import albumentations as A

class SegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, augment=True):
        self.imgs = img_paths
        self.masks = mask_paths
        self.augment = augment

        self.class_map = {
            100: 0,   # Trees
            200: 1,   # Lush Bushes
            300: 2,   # Dry Grass
            500: 3,   # Dry Bushes
            550: 4,   # Ground Clutter
            600: 5,   # Flowers
            700: 6,   # Logs
            800: 7,   # Rocks
            7100: 8,  # Landscape
            10000: 9  # Sky
        }

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20,
                p=0.7
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.6
            ),
            A.GaussianBlur(p=0.2),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 64), hole_width_range=(8, 64), p=0.4),
            A.Resize(512, 512),
        ])

    def __len__(self):
        return len(self.imgs)

    def remap_mask(self, mask):
        remapped = np.zeros_like(mask, dtype=np.int64) + 8
        for k, v in self.class_map.items():
            remapped[mask == k] = v
        return remapped

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.imgs[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks[idx], 0)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {self.masks[idx]}")

        if self.augment:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        else:
            val_transform = A.Resize(512, 512)
            augmented = val_transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        label_mask = self.remap_mask(mask)
            
        return img, label_mask