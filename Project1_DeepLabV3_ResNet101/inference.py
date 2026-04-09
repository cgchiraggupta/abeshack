import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import albumentations as A
from models.deeplabv3plus import get_model

# ---------------- CONFIG ----------------
MODEL_PATH = "checkpoints/best_model.pth"
TEST_IMG_DIR = "data/testImages/Color_Images"
SAVE_DIR = "inference_results"
IMG_SIZE = 512
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- COLORS (RGB) ----------------
# Standardized colors for Duality AI classes (mapped 0-9)
# Mapping: 0:Trees, 1:Lush Bushes, 2:Dry Grass, 3:Dry Bushes, 4:Ground Clutter, 
# 5:Flowers, 6:Logs, 7:Rocks, 8:Landscape, 9:Sky
COLORS = np.array([
    [34, 139, 34],    # 0: Trees
    [0, 255, 0],      # 1: Lush Bushes
    [189, 183, 107],  # 2: Dry Grass
    [160, 82, 45],    # 3: Dry Bushes
    [105, 105, 105],  # 4: Ground Clutter
    [255, 0, 255],    # 5: Flowers
    [139, 69, 19],    # 6: Logs
    [128, 128, 128],  # 7: Rocks
    [210, 180, 140],  # 8: Landscape (Catch-all)
    [135, 206, 235],  # 9: Sky
], dtype=np.uint8)

# ---------------- MODEL ----------------
print(f"Loading model from {MODEL_PATH}...")
model = get_model(NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------- TRANSFORM ----------------
# Use same transforms as validation (Resize + Normalization)
val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

# ---------------- INFERENCE ----------------
print(f"Starting inference on {TEST_IMG_DIR}...")
images = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in tqdm(images):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    image_bgr = cv2.imread(img_path)
    if image_bgr is None: continue
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    aug = val_transform(image=image_rgb)
    img_resized = aug['image']
    
    # To Tensor format (matching training)
    input_tensor = img_resized.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            output = model(input_tensor)
            # Normalization
            output = torch.softmax(output, dim=1)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Morphological Cleaning (Removes small noise blobs)
    kernel = np.ones((5, 5), np.uint8)
    # Convert to unit8 for morphology, then back if needed
    # (Note: for multiclass, we apply cleaning to each class mask if needed, 
    # but here we do a simple pass on the full prediction)
    pred = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Colorize mask
    color_mask_rgb = COLORS[pred]

    # Create Overlay (on the resized 512x512 image for consistency in side-by-side)
    overlay_rgb = cv2.addWeighted(img_resized, 0.6, color_mask_rgb, 0.4, 0)

    # Convert to BGR for saving
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    img_resized_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    # ---------------- SIDE-BY-SIDE VISUAL ----------------
    # [ Original | Predicted Mask | Overlay ]
    combined = np.hstack((img_resized_bgr, color_mask_bgr, overlay_bgr))
    
    # Add textual labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Predicted Mask", (IMG_SIZE + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Overlay", (IMG_SIZE*2 + 10, 30), font, 0.8, (255, 255, 255), 2)

    save_path = os.path.join(SAVE_DIR, f"result_{img_name}")
    cv2.imwrite(save_path, combined)
    
    # Debug: Save raw index mask scaled for visibility if needed
    # cv2.imwrite(os.path.join(SAVE_DIR, f"debug_{img_name}"), pred * 25)

print(f"\nInference complete! Results saved to '{SAVE_DIR}/' folder.")
print("Each image shows: [ Original | Predicted Mask | Overlay ]")
