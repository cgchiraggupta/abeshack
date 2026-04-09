import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

class OffRoadDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, test_mode=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.test_mode = test_mode
        
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        assert len(self.image_files) == len(self.mask_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        
        self.class_mapping = {
            100: 1,   # Trees
            200: 2,   # Lush Bushes
            300: 3,   # Dry Bushes
            400: 4,   # Grass
            500: 5,   # Concrete
            600: 6,   # Rocks
            700: 7,   # Water
            800: 8,   # Dirt
            900: 9,   # Mud
            1000: 10  # Snow
        }
        
        print(f"Dataset initialized with {len(self.image_files)} samples")
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def process_mask(self, mask):
        processed_mask = np.zeros_like(mask, dtype=np.uint8)
        
        for original_value, mapped_value in self.class_mapping.items():
            processed_mask[mask == original_value] = mapped_value
        
        return processed_mask
    
    def augment_image_mask(self, image, mask):
        if self.transform is None:
            return image, mask
        
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask']
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            raise ValueError(f"Could not read mask at {mask_path}")
        
        mask = self.process_mask(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask

class OffRoadTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                  if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        
        print(f"Test dataset initialized with {len(self.image_files)} images")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, self.image_files[idx], original_size

class OffRoadDatasetWithWeights(OffRoadDataset):
    def __init__(self, image_dir, mask_dir, transform=None, class_weights=None):
        super().__init__(image_dir, mask_dir, transform)
        self.class_weights = class_weights
    
    def __getitem__(self, idx):
        image, mask = super().__getitem__(idx)
        
        if self.class_weights is not None:
            weight_map = torch.zeros_like(mask, dtype=torch.float32)
            for class_idx, weight in enumerate(self.class_weights):
                weight_map[mask == class_idx] = weight
            return image, mask, weight_map
        
        return image, mask

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5)
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_test_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def compute_class_weights(mask_dir, num_classes=11):
    import glob
    from tqdm import tqdm
    
    mask_files = glob.glob(os.path.join(mask_dir, '*.png'))
    if not mask_files:
        mask_files = glob.glob(os.path.join(mask_dir, '*.jpg'))
    
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    print("Computing class weights from masks...")
    for mask_file in tqdm(mask_files):
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCAPE)
        
        unique_values = np.unique(mask)
        for val in unique_values:
            class_idx = val // 100
            if class_idx < num_classes:
                class_counts[class_idx] += np.sum(mask == val)
    
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    
    median_freq = np.median(class_frequencies[class_frequencies > 0])
    class_weights = median_freq / class_frequencies
    class_weights[class_frequencies == 0] = 0
    
    print("Class frequencies:", class_frequencies)
    print("Class weights:", class_weights)
    
    return torch.tensor(class_weights, dtype=torch.float32)

def visualize_dataset_samples(dataset, num_samples=4):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 5))
    
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, mask = dataset[idx]
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy()
            if image.shape[2] == 3:
                image = (image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Sample {idx} - Image")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask, cmap='tab20')
        axes[i, 1].set_title(f"Sample {idx} - Mask")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    train_dataset = OffRoadDataset(
        image_dir='data/train/Color_Images',
        mask_dir='data/train/Segmentation',
        transform=get_train_transforms()
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    val_dataset = OffRoadDataset(
        image_dir='data/val/Color_Images',
        mask_dir='data/val/Segmentation',
        transform=get_val_transforms()
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    sample_image, sample_mask = train_dataset[0]
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample mask shape: {sample_mask.shape}")
    print(f"Unique mask values: {torch.unique(sample_mask)}")
    
    class_weights = compute_class_weights('data/train/Segmentation')
    print(f"Computed class weights: {class_weights}")