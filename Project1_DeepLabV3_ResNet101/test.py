import torch
import os
import glob
import cv2
import numpy as np
from models.deeplabv3plus import get_model
from dataset.dataset import SegDataset
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score
from tqdm import tqdm

def evaluate(mode="val"):
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path to best model
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    # Load model
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Data paths
    if mode == "val":
        img_dir = "data/val/Color_Images"
        mask_dir = "data/val/Segmentation"
    else:
        img_dir = "data/testImages/Color_Images"
        mask_dir = "data/testImages/Segmentation"

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    masks_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if not imgs:
        print(f"Warning: No {mode} images found in {img_dir}.")
        return

    ds = SegDataset(imgs, masks_files, augment=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    all_preds = []
    all_masks = []

    with torch.no_grad():
        for i_imgs, i_masks in tqdm(loader, desc=f"Evaluating {mode}"):
            i_imgs = i_imgs.to(DEVICE)
            outputs = model(i_imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.append(preds.flatten())
            all_masks.append(i_masks.numpy().flatten())

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Calculate IoU per class
    iou_scores = jaccard_score(all_masks, all_preds, average=None, labels=list(range(NUM_CLASSES)))
    mean_iou = np.mean(iou_scores)

    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]

    print("\n--- Model Performance Evaluation ---")
    for i, score in enumerate(iou_scores):
        print(f"{class_names[i]:<15}: {score:.4f}")
    
    print(f"\nMean IoU: {mean_iou:.4f}")

    # Visualization Colors
    class_colors = np.array([
        [0, 255, 0],    # Trees
        [0, 128, 0],    # Lush Bushes
        [255, 255, 0],  # Dry Grass
        [128, 128, 0],  # Dry Bushes
        [139, 69, 19],  # Ground Clutter
        [255, 105, 180],# Flowers
        [160, 82, 45],  # Logs
        [128, 128, 128],# Rocks
        [210, 180, 140],# Landscape
        [135, 206, 235] # Sky
    ], dtype=np.uint8)

    print("\nSaving visualization samples to 'results/'...")
    for i in range(min(10, len(imgs))):
        img_path = imgs[i]
        mask_path = masks_files[i]
        
        # Original image
        img_orig = cv2.imread(img_path)
        img_vis = cv2.resize(img_orig, (512, 512))
        
        # Load and process through model
        batch_img, batch_mask = ds[i]
        input_tensor = torch.from_numpy(batch_img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            
        # Create color masks
        gt_color = class_colors[batch_mask]
        pred_color = class_colors[pred]

        # Convert back to BGR for CV2
        gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
        pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)

        # Concatenate: Image | Ground Truth | Prediction
        combined = np.hstack((img_vis, gt_color, pred_color))
        
        # Add labels
        cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Ground Truth", (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, "Prediction", (1034, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        save_path = f"results/sample_{i}_{mode}.png"
        cv2.imwrite(save_path, combined)
        
    print(f"\nEvaluation on {mode} complete. Performance details printed above.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="val", choices=["val", "test"])
    args = parser.parse_args()
    
    evaluate(mode=args.mode)
