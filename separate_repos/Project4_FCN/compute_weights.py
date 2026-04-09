import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_class_weights():
    train_mask_dir = "data/train/Segmentation"
    val_mask_dir = "data/val/Segmentation"
    
    class_map = {
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
    
    train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))
    
    if not train_masks:
        print(f"Error: No training masks found in {train_mask_dir}")
        return
    
    print(f"Found {len(train_masks)} training masks and {len(val_masks)} validation masks")
    
    class_counts = np.zeros(10, dtype=np.float64)
    total_pixels = 0
    
    print("Analyzing training masks...")
    for mask_path in tqdm(train_masks):
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            continue
        
        for original_id, mapped_id in class_map.items():
            class_counts[mapped_id] += np.sum(mask == original_id)
        
        total_pixels += mask.size
    
    print("\nClass Distribution Analysis:")
    print("="*50)
    
    class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
                   "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
    
    for i, (name, count) in enumerate(zip(class_names, class_counts)):
        percentage = (count / total_pixels) * 100
        print(f"{name:15s}: {count:12,.0f} pixels ({percentage:6.2f}%)")
    
    print(f"\nTotal pixels analyzed: {total_pixels:,}")
    
    class_frequencies = class_counts / total_pixels
    median_freq = np.median(class_frequencies[class_frequencies > 0])
    
    print(f"\nMedian class frequency: {median_freq:.6f}")
    
    weights = median_freq / class_frequencies
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    
    print("\nComputed Class Weights (inverse frequency):")
    print("="*50)
    for i, (name, weight) in enumerate(zip(class_names, weights)):
        print(f"{name:15s}: {weight:10.4f}")
    
    print("\nWeights array for training:")
    print(f"[{', '.join([f'{w:.4f}' for w in weights])}]")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(10), class_counts / total_pixels * 100)
    plt.xlabel('Class')
    plt.ylabel('Percentage (%)')
    plt.title('Class Distribution')
    plt.xticks(range(10), [name[:3] for name in class_names], rotation=45)
    
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count/total_pixels*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), weights)
    plt.xlabel('Class')
    plt.ylabel('Weight')
    plt.title('Class Weights (Inverse Frequency)')
    plt.xticks(range(10), [name[:3] for name in class_names], rotation=45)
    
    for i, weight in enumerate(weights):
        plt.text(i, weight + 0.5, f'{weight:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('class_distribution_analysis.png', dpi=150)
    plt.show()
    
    print(f"\nAnalysis saved to class_distribution_analysis.png")
    
    return weights.tolist()

def check_dataset_leakage():
    train_img_dir = "data/train/Color_Images"
    val_img_dir = "data/val/Color_Images"
    
    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    
    train_hashes = []
    val_hashes = []
    
    print("Computing image hashes to check for leakage...")
    
    for img_path in tqdm(train_images):
        img = cv2.imread(img_path)
        if img is not None:
            img_hash = hash(img.tobytes())
            train_hashes.append(img_hash)
    
    for img_path in tqdm(val_images):
        img = cv2.imread(img_path)
        if img is not None:
            img_hash = hash(img.tobytes())
            val_hashes.append(img_hash)
    
    train_set = set(train_hashes)
    val_set = set(val_hashes)
    
    intersection = train_set.intersection(val_set)
    
    print(f"\nDataset Leakage Check:")
    print("="*50)
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Unique training images: {len(train_set)}")
    print(f"Unique validation images: {len(val_set)}")
    print(f"Images in both sets: {len(intersection)}")
    
    if len(intersection) > 0:
        print(f"⚠️  WARNING: {len(intersection)} images appear in both training and validation sets!")
        print("This may cause over-optimistic validation metrics.")
    else:
        print("✅ No leakage detected between training and validation sets.")
    
    return len(intersection) == 0

if __name__ == "__main__":
    print("Off-Road Terrain Segmentation - Class Weight Analysis")
    print("="*60)
    
    weights = compute_class_weights()
    
    print("\n" + "="*60)
    print("Dataset Integrity Check")
    print("="*60)
    
    leakage_check = check_dataset_leakage()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("1. Copy the weights array above into your training script")
    print("2. Ensure no dataset leakage exists")
    print("3. Use these weights in CombinedLoss for balanced training")
    
    if weights:
        print(f"\nRecommended weights for training:")
        print(f"class_weights = torch.tensor({weights})")