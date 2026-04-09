import os
import hashlib
from tqdm import tqdm

def get_image_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def load_images(folder):
    images = {}
    if not os.path.exists(folder):
        print(f"Warning: {folder} does not exist.")
        return images
        
    print(f"Hashing images in {folder}...")
    for img in tqdm(os.listdir(folder)):
        path = os.path.join(folder, img)
        if os.path.isfile(path) and img.lower().endswith(('.png', '.jpg', '.jpeg')):
            images[img] = get_image_hash(path)
    return images

# Corrected paths matching our project structure
train_dir = "data/train/Color_Images"
val_dir   = "data/val/Color_Images"
test_dir  = "data/testImages/Color_Images"

train_imgs = load_images(train_dir)
val_imgs   = load_images(val_dir)
test_imgs  = load_images(test_dir)

# Filename overlap
print("\n--- Filename Overlap Analysis ---")
print("Train-Val overlap:", set(train_imgs) & set(val_imgs))
print("Train-Test overlap:", set(train_imgs) & set(test_imgs))
print("Val-Test overlap:", set(val_imgs) & set(test_imgs))

# Hash overlap
print("\n--- Content Overlap Analysis (Hash-based) ---")

train_hashes = set(train_imgs.values())
val_hashes   = set(val_imgs.values())
test_hashes  = set(test_imgs.values())

print("Train-Val overlap:", train_hashes & val_hashes)
print("Train-Test overlap:", train_hashes & test_hashes)
print("Val-Test overlap:", val_hashes & test_hashes)

# Integrity Declaration
print("\n--- Integrity Summary ---")
if not (train_hashes & val_hashes) and not (train_hashes & test_hashes) and not (val_hashes & test_hashes):
    print("✅ SUCCESS: No content leakage detected across splits.")
    print("   Data splits are independent and clean for hackathon evaluation.")
else:
    print("🚨 WARNING: Content overlap detected! Check previous results.")
