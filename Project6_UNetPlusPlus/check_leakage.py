import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import hashlib

def compute_image_hash(image_path):
    """Compute MD5 hash of an image file"""
    with open(image_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    return file_hash

def compute_pixel_hash(image_path):
    """Compute hash based on pixel values (slower but more accurate)"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    return hashlib.md5(img.tobytes()).hexdigest()

def check_leakage_between_splits():
    """Check for duplicate images between train and validation splits"""
    print("Checking for dataset leakage between train and validation splits...")
    
    train_img_dir = "data/train/Color_Images"
    val_img_dir = "data/val/Color_Images"
    
    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    
    if not train_images or not val_images:
        print("Error: Could not find images in train or validation directories")
        return False
    
    print(f"Found {len(train_images)} training images and {len(val_images)} validation images")
    
    train_hashes = {}
    val_hashes = {}
    
    print("Computing file hashes for training images...")
    for img_path in tqdm(train_images):
        file_hash = compute_image_hash(img_path)
        train_hashes[file_hash] = img_path
    
    print("Computing file hashes for validation images...")
    for img_path in tqdm(val_images):
        file_hash = compute_image_hash(img_path)
        val_hashes[file_hash] = img_path
    
    common_hashes = set(train_hashes.keys()).intersection(set(val_hashes.keys()))
    
    if common_hashes:
        print(f"\n⚠️  WARNING: Found {len(common_hashes)} duplicate images between train and validation sets!")
        print("Duplicate files:")
        for hash_val in common_hashes:
            print(f"  - Train: {train_hashes[hash_val]}")
            print(f"    Val:   {val_hashes[hash_val]}")
        return False
    else:
        print("✅ No file-level duplicates found between train and validation sets")
        return True

def check_duplicates_within_split(split_dir, split_name="train"):
    """Check for duplicate images within a single split"""
    print(f"\nChecking for duplicates within {split_name} split...")
    
    images = sorted(glob.glob(os.path.join(split_dir, "*.png")))
    
    if not images:
        print(f"Error: No images found in {split_dir}")
        return True
    
    print(f"Found {len(images)} images in {split_name} split")
    
    hashes = {}
    duplicates = []
    
    print(f"Computing hashes for {split_name} images...")
    for img_path in tqdm(images):
        file_hash = compute_image_hash(img_path)
        if file_hash in hashes:
            duplicates.append((img_path, hashes[file_hash]))
        else:
            hashes[file_hash] = img_path
    
    if duplicates:
        print(f"\n⚠️  WARNING: Found {len(duplicates)} duplicate images within {split_name} split!")
        print("Duplicate pairs:")
        for dup1, dup2 in duplicates[:10]:  # Show first 10 duplicates
            print(f"  - {dup1}")
            print(f"    {dup2}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicates")
        return False
    else:
        print(f"✅ No duplicates found within {split_name} split")
        return True

def check_mask_alignment():
    """Check that each image has a corresponding mask"""
    print("\nChecking image-mask alignment...")
    
    train_img_dir = "data/train/Color_Images"
    train_mask_dir = "data/train/Segmentation"
    val_img_dir = "data/val/Color_Images"
    val_mask_dir = "data/val/Segmentation"
    
    train_images = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")))
    val_images = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))
    
    train_image_names = {os.path.basename(p) for p in train_images}
    train_mask_names = {os.path.basename(p) for p in train_masks}
    val_image_names = {os.path.basename(p) for p in val_images}
    val_mask_names = {os.path.basename(p) for p in val_masks}
    
    train_missing_masks = train_image_names - train_mask_names
    train_extra_masks = train_mask_names - train_image_names
    val_missing_masks = val_image_names - val_mask_names
    val_extra_masks = val_mask_names - val_image_names
    
    issues_found = False
    
    if train_missing_masks:
        print(f"⚠️  WARNING: {len(train_missing_masks)} training images missing masks:")
        for mask in sorted(train_missing_masks)[:5]:
            print(f"  - {mask}")
        if len(train_missing_masks) > 5:
            print(f"  ... and {len(train_missing_masks) - 5} more")
        issues_found = True
    
    if train_extra_masks:
        print(f"⚠️  WARNING: {len(train_extra_masks)} training masks without corresponding images:")
        for mask in sorted(train_extra_masks)[:5]:
            print(f"  - {mask}")
        if len(train_extra_masks) > 5:
            print(f"  ... and {len(train_extra_masks) - 5} more")
        issues_found = True
    
    if val_missing_masks:
        print(f"⚠️  WARNING: {len(val_missing_masks)} validation images missing masks:")
        for mask in sorted(val_missing_masks)[:5]:
            print(f"  - {mask}")
        if len(val_missing_masks) > 5:
            print(f"  ... and {len(val_missing_masks) - 5} more")
        issues_found = True
    
    if val_extra_masks:
        print(f"⚠️  WARNING: {len(val_extra_masks)} validation masks without corresponding images:")
        for mask in sorted(val_extra_masks)[:5]:
            print(f"  - {mask}")
        if len(val_extra_masks) > 5:
            print(f"  ... and {len(val_extra_masks) - 5} more")
        issues_found = True
    
    if not issues_found:
        print("✅ All images have corresponding masks")
    
    return not issues_found

def check_mask_classes():
    """Check that masks contain valid class values"""
    print("\nChecking mask class values...")
    
    train_mask_dir = "data/train/Segmentation"
    val_mask_dir = "data/val/Segmentation"
    
    valid_classes = {100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000}
    
    issues_found = False
    
    for split_name, mask_dir in [("train", train_mask_dir), ("val", val_mask_dir)]:
        masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        
        if not masks:
            print(f"Error: No masks found in {mask_dir}")
            continue
        
        print(f"Checking {len(masks)} {split_name} masks...")
        
        for mask_path in tqdm(masks[:100]):  # Check first 100 masks
            mask = cv2.imread(mask_path, 0)
            if mask is None:
                print(f"  ⚠️  Could not read mask: {mask_path}")
                issues_found = True
                continue
            
            unique_values = np.unique(mask)
            invalid_values = [v for v in unique_values if v not in valid_classes and v != 0]
            
            if invalid_values:
                print(f"  ⚠️  Invalid class values in {os.path.basename(mask_path)}: {invalid_values}")
                issues_found = True
    
    if not issues_found:
        print("✅ All masks contain valid class values")
    
    return not issues_found

def main():
    print("Off-Road Terrain Segmentation - Dataset Integrity Check")
    print("="*70)
    
    all_checks_passed = True
    
    check1 = check_leakage_between_splits()
    all_checks_passed = all_checks_passed and check1
    
    check2 = check_duplicates_within_split("data/train/Color_Images", "train")
    all_checks_passed = all_checks_passed and check2
    
    check3 = check_duplicates_within_split("data/val/Color_Images", "validation")
    all_checks_passed = all_checks_passed and check3
    
    check4 = check_mask_alignment()
    all_checks_passed = all_checks_passed and check4
    
    check5 = check_mask_classes()
    all_checks_passed = all_checks_passed and check5
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_checks_passed:
        print("✅ All dataset integrity checks passed!")
        print("Your dataset is ready for training.")
    else:
        print("⚠️  Some issues were found in the dataset.")
        print("Please fix these issues before training to ensure reliable results.")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)