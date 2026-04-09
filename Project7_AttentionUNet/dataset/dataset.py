import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OffRoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=True, image_size=512):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) and masks ({len(self.mask_files)}) don't match"
        
        self.class_mapping = {
            100: 0,   # Trees
            200: 1,   # Lush Bushes
            300: 2,   # Dry Bushes
            400: 3,   # Grass
            500: 4,   # Dirt
            600: 5,   # Gravel
            700: 6,   # Rocks
            800: 7,   # Sand
            900: 8,   # Water
            1000: 9   # Sky
        }
        
        self.augmentations = A.Compose([
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        self.val_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        mask_processed = np.zeros_like(mask_np, dtype=np.uint8)
        
        for original_value, mapped_value in self.class_mapping.items():
            mask_processed[mask_np == original_value] = mapped_value
        
        if self.transform:
            augmented = self.augmentations(image=image_np, mask=mask_processed)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].long()
        else:
            augmented = self.val_transform(image=image_np, mask=mask_processed)
            image_tensor = augmented['image']
            mask_tensor = augmented['mask'].long()
        
        return image_tensor, mask_tensor
    
    def get_class_distribution(self):
        class_counts = {i: 0 for i in range(len(self.class_mapping))}
        
        for idx in range(len(self)):
            _, mask = self[idx]
            unique, counts = torch.unique(mask, return_counts=True)
            
            for class_idx, count in zip(unique.tolist(), counts.tolist()):
                if class_idx in class_counts:
                    class_counts[class_idx] += count
        
        total_pixels = sum(class_counts.values())
        class_weights = {}
        
        for class_idx, count in class_counts.items():
            if count > 0:
                class_weights[class_idx] = total_pixels / (len(class_counts) * count)
            else:
                class_weights[class_idx] = 1.0
        
        return class_counts, class_weights
    
    def visualize_sample(self, idx, save_path=None):
        import matplotlib.pyplot as plt
        
        image, mask = self[idx]
        
        image_np = image.numpy().transpose(1, 2, 0)
        image_np = image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=len(self.class_mapping)-1)
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return image_np, mask_np

if __name__ == "__main__":
    # Test the dataset
    print("Testing OffRoadDataset...")
    
    # Update these paths according to your actual data structure
    test_image_dir = "data/train/Color_Images"
    test_mask_dir = "data/train/Segmentation"
    
    if os.path.exists(test_image_dir) and os.path.exists(test_mask_dir):
        dataset = OffRoadDataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            transform=True
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Unique values in mask: {torch.unique(mask)}")
            
            class_counts, class_weights = dataset.get_class_distribution()
            print(f"\nClass distribution:")
            for class_idx, count in class_counts.items():
                print(f"  Class {class_idx}: {count} pixels")
            
            print(f"\nClass weights (for loss function):")
            for class_idx, weight in class_weights.items():
                print(f"  Class {class_idx}: {weight:.4f}")
        else:
            print("No images found in the dataset directories.")
    else:
        print(f"Test directories not found. Please update the paths in dataset.py")
        print(f"Image dir exists: {os.path.exists(test_image_dir)}")
        print(f"Mask dir exists: {os.path.exists(test_mask_dir)}")