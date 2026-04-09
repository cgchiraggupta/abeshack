import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random

class OffRoadDataset(Dataset):
    """
    Dataset for off-road terrain semantic segmentation
    """
    def __init__(self, data_dir, split='train', transform=None, debug=False):
        """
        Args:
            data_dir: Root directory of dataset
            split: 'train', 'val', or 'test'
            transform: Albumentations transforms
            debug: Enable debug mode for printing info
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.debug = debug
        
        # Class mapping (from the original project)
        self.class_mapping = {
            100: 0,  # Trees
            200: 1,  # Lush Bushes
            300: 2,  # Grass
            400: 3,  # Dirt
            500: 4,  # Sand
            600: 5,  # Water
            700: 6,  # Rocks
            800: 7,  # Bushes
            900: 8,  # Mud
            0:   9   # Background
        }
        
        # Class names
        self.class_names = [
            'Trees', 'Lush Bushes', 'Grass', 'Dirt', 'Sand',
            'Water', 'Rocks', 'Bushes', 'Mud', 'Background'
        ]
        
        # Class colors for visualization
        self.class_colors = [
            (34, 139, 34),    # Trees - Forest Green
            (0, 100, 0),      # Lush Bushes - Dark Green
            (124, 252, 0),    # Grass - Lawn Green
            (139, 69, 19),    # Dirt - Saddle Brown
            (238, 203, 173),  # Sand - Burlywood
            (30, 144, 255),   # Water - Dodger Blue
            (128, 128, 128),  # Rocks - Gray
            (0, 128, 0),      # Bushes - Green
            (101, 67, 33),    # Mud - Dark Brown
            (0, 0, 0)         # Background - Black
        ]
        
        # Load data paths
        self.image_paths = []
        self.mask_paths = []
        
        self._load_data_paths()
        
        if self.debug:
            print(f"{split} dataset loaded:")
            print(f"  Number of images: {len(self.image_paths)}")
            print(f"  Number of masks: {len(self.mask_paths)}")
            if len(self.image_paths) > 0:
                print(f"  Sample image path: {self.image_paths[0]}")
                print(f"  Sample mask path: {self.mask_paths[0]}")
    
    def _load_data_paths(self):
        """Load image and mask paths based on split"""
        if self.split == 'train':
            image_dir = os.path.join(self.data_dir, 'train', 'Color_Images')
            mask_dir = os.path.join(self.data_dir, 'train', 'Segmentation')
        elif self.split == 'val':
            image_dir = os.path.join(self.data_dir, 'val', 'Color_Images')
            mask_dir = os.path.join(self.data_dir, 'val', 'Segmentation')
        elif self.split == 'test':
            image_dir = os.path.join(self.data_dir, 'testImages')
            mask_dir = None  # Test may not have masks
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Check if directories exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        for ext in image_extensions:
            self.image_paths.extend(self._find_files(image_dir, ext))
            self.image_paths.extend(self._find_files(image_dir, ext.upper()))
        
        # Sort to ensure consistent ordering
        self.image_paths.sort()
        
        # For train and val, find corresponding masks
        if self.split in ['train', 'val'] and mask_dir and os.path.exists(mask_dir):
            for img_path in self.image_paths:
                # Get base filename without extension
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                
                # Look for mask with same name (could be .png, .jpg, etc.)
                mask_found = False
                for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                    mask_path = os.path.join(mask_dir, base_name + ext)
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)
                        mask_found = True
                        break
                
                if not mask_found:
                    # Try with different naming conventions
                    mask_path = os.path.join(mask_dir, base_name + '_mask.png')
                    if os.path.exists(mask_path):
                        self.mask_paths.append(mask_path)
                    else:
                        # If no mask found, remove the image from list
                        self.image_paths.remove(img_path)
                        if self.debug:
                            print(f"Warning: No mask found for {img_path}")
        
        elif self.split == 'test':
            # For test, create dummy mask paths
            self.mask_paths = [None] * len(self.image_paths)
    
    def _find_files(self, directory, extension):
        """Find files with given extension in directory"""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith(extension):
                    files.append(os.path.join(root, filename))
        return files
    
    def _load_image(self, image_path):
        """Load and preprocess image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def _load_mask(self, mask_path):
        """Load and preprocess mask"""
        if mask_path is None:
            # For test images without masks, return empty mask
            return np.zeros((512, 512), dtype=np.uint8)
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        # Apply class mapping
        mapped_mask = np.zeros_like(mask, dtype=np.uint8)
        for original_value, mapped_value in self.class_mapping.items():
            mapped_mask[mask == original_value] = mapped_value
        
        return mapped_mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get item at index
        
        Returns:
            image: Transformed image tensor
            mask: Transformed mask tensor
            image_path: Path to original image (for debugging)
        """
        try:
            # Load image and mask
            image_path = self.image_paths[idx]
            mask_path = self.mask_paths[idx]
            
            image = self._load_image(image_path)
            mask = self._load_mask(mask_path)
            
            # Apply transformations if specified
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                # Default transform: convert to tensor
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                mask = torch.from_numpy(mask).long()
            
            return image, mask, image_path
        
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            print(f"Image path: {self.image_paths[idx]}")
            print(f"Mask path: {self.mask_paths[idx]}")
            raise
    
    def get_class_distribution(self):
        """
        Calculate class distribution in the dataset
        
        Returns:
            class_counts: Dictionary with count for each class
            class_weights: Tensor of class weights for loss function
        """
        class_counts = torch.zeros(len(self.class_names))
        
        for _, mask, _ in self:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = mask
            
            # Count each class
            for class_idx in range(len(self.class_names)):
                class_counts[class_idx] += (mask_np == class_idx).sum()
        
        # Calculate weights (inverse frequency)
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (len(self.class_names) * class_counts + 1e-8)
        
        # Normalize weights
        class_weights = class_weights / class_weights.sum()
        
        return class_counts, class_weights
    
    def visualize_sample(self, idx=0, save_path=None):
        """
        Visualize a sample from the dataset
        
        Args:
            idx: Index of sample to visualize
            save_path: Path to save visualization (optional)
        """
        import matplotlib.pyplot as plt
        
        # Get sample
        image, mask, image_path = self[idx]
        
        # Convert tensors to numpy for visualization
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).numpy()
            # Denormalize if normalized
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
        
        if isinstance(mask, torch.Tensor):
            mask_np = mask.numpy()
        else:
            mask_np = mask
        
        # Create colored mask
        colored_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.class_colors):
            colored_mask[mask_np == class_idx] = color
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title(f'Image\n{os.path.basename(image_path)}')
        axes[0].axis('off')
        
        axes[1].imshow(colored_mask)
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        
        # Create overlay
        overlay = cv2.addWeighted(image_np, 0.5, colored_mask, 0.5, 0)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (50% transparency)')
        axes[2].axis('off')
        
        plt.suptitle(f'Sample {idx} - {self.split} set', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        # Print class distribution for this sample
        unique, counts = np.unique(mask_np, return_counts=True)
        print(f"\nClass distribution for sample {idx}:")
        for class_idx, count in zip(unique, counts):
            class_name = self.class_names[class_idx]
            percentage = (count / mask_np.size) * 100
            print(f"  {class_name}: {count} pixels ({percentage:.1f}%)")
    
    def get_class_colors_tensor(self):
        """Get class colors as tensor for visualization"""
        colors = torch.tensor(self.class_colors, dtype=torch.float32) / 255.0
        return colors

def get_transforms(split='train', image_size=512):
    """
    Get Albumentations transforms for dataset
    
    Args:
        split: 'train', 'val', or 'test'
        image_size: Size to resize images to
    
    Returns:
        transform: Albumentations transform
    """
    if split == 'train':
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.Blur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    elif split in ['val', 'test']:
        transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    else:
        raise ValueError(f"Invalid split: {split}")
    
    return transform

def create_dataloaders(data_dir, batch_size=4, num_workers=4, image_size=512, debug=False):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory of dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Size to resize images to
        debug: Enable debug mode
    
    Returns:
        train_loader, val_loader, test_loader: DataLoader objects
        class_weights: Class weights for loss function
    """
    # Get transforms
    train_transform = get_transforms('train', image_size)
    val_transform = get_transforms('val', image_size)
    test_transform = get_transforms('test', image_size)
    
    # Create datasets
    train_dataset = OffRoadDataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform,
        debug=debug
    )
    
    val_dataset = OffRoadDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform,
        debug=debug
    )
    
    test_dataset = OffRoadDataset(
        data_dir=data_dir,
        split='test',
        transform=test_transform,
        debug=debug
    )
    
    # Calculate class weights from training set
    _, class_weights = train_dataset.get_class_distribution()
    
    if debug:
        print(f"\nDataset sizes:")
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        print(f"\nClass weights:")
        for i, (name, weight) in enumerate(zip(train_dataset.class_names, class_weights)):
            print(f"  {name}: {weight:.4f}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights

def compute_class_weights(data_dir, split='train'):
    """
    Compute class weights for the dataset
    
    Args:
        data_dir: Root directory of dataset
        split: Which split to compute weights from
    
    Returns:
        class_weights: Tensor of class weights
    """
    dataset = OffRoadDataset(data_dir=data_dir, split=split, transform=None)
    _, class_weights = dataset.get_class_distribution()
    
    return class_weights

# Example usage
if __name__ == "__main__":
    # Test the dataset
    data_dir = "data"  # Update this path
    
    try:
        # Create a sample dataset
        print("Testing OffRoadDataset...")
        
        # Test with debug mode
        train_dataset = OffRoadDataset(
            data_dir=data_dir,
            split='train',
            transform=None,
            debug=True
        )
        
        if len(train_dataset) > 0:
            # Visualize first sample
            train_dataset.visualize_sample(idx=0)
            
            # Get class distribution
            class_counts, class_weights = train_dataset.get_class_distribution()
            
            print("\nClass distribution in training set:")
            for i, (name, count, weight) in enumerate(zip(train_dataset.class_names, class_counts, class_weights)):
                print(f"  {name}: {count:.0f} pixels, weight: {weight:.4f}")
        
        # Test with transforms
        print("\nTesting with transforms...")
        transform = get_transforms('train', image_size=512)
        train_dataset_transformed = OffRoadDataset(
            data_dir=data_dir,
            split='train',
            transform=transform,
            debug=False
        )
        
        if len(train_dataset_transformed) > 0:
            image, mask, path = train_dataset_transformed[0]
            print(f"Transformed image shape: {image.shape}")
            print(f"Transformed mask shape: {mask.shape}")
            print(f"Image dtype: {image.dtype}")
            print(f"Mask dtype: {mask.dtype}")
        
        # Test dataloader creation
        print("\nTesting dataloader creation...")
        train_loader, val_loader, test_loader, weights = create_dataloaders(
            data_dir=data_dir,
            batch_size=2,
            num_workers=0,
            debug=True
        )
        
        # Test batch loading
        if train_loader:
            print("\nTesting batch loading...")
            for batch_idx, (images, masks, paths) in enumerate(train_loader):
                print(f"Batch {batch_idx}:")
                print(f"  Images shape: {images.shape}")
                print(f"  Masks shape: {masks.shape}")
                print(f"  Unique mask values: {torch.unique(masks)}")
                
                if batch_idx >= 2:  # Just test first 3 batches
                    break
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please update the data_dir path in the test code.")
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()