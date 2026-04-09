import os
import torch
import numpy as np
from tqdm import tqdm
import cv2

MASK_DIR = "data/train/Segmentation"
NUM_CLASSES = 10

# Map original IDs to 0-9
CLASS_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4, 
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

def compute_weights():
    print(f"Analyzing pixel frequencies in {MASK_DIR}...")
    class_counts = np.zeros(NUM_CLASSES)
    
    mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith('.png')]
    print(f"Total files found: {len(mask_files)}")
    
    for mask_file in tqdm(mask_files):
        mask_path = os.path.join(MASK_DIR, mask_file)
        # Using cv2 to match dataset.py behavior for these high-value masks
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None: continue
        
        # Count each unique value
        uniques, counts = np.unique(mask, return_counts=True)
        
        for val, count in zip(uniques, counts):
            if val in CLASS_MAP:
                class_counts[CLASS_MAP[val]] += count
            else:
                # Default unknown to Landscape
                class_counts[8] += count
        
    total_pixels = class_counts.sum()
    if total_pixels == 0:
        print("Error: No pixels found!")
        return

    class_freq = class_counts / total_pixels
    
    # Avoid log(0)
    eps = 1e-6
    weights = 1.0 / np.log(1.02 + class_freq + eps)
    
    print("\n--- Statistics ---")
    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    for i, name in enumerate(class_names):
        print(f"{name}: {class_freq[i]:.4%} (Weight: {weights[i]:.4f})")
        
    print("\nCopy this tensor to your train.py:")
    print(f"class_weights = torch.tensor({list(np.round(weights, 4))})")

if __name__ == "__main__":
    compute_weights()
