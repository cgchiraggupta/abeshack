import torch
import os
import glob
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset.dataset import SegDataset
from metrics import dice_score, iou_score

def load_model(num_classes=10, device="cuda"):
    from models.fcn import get_model
    model_path = "checkpoints/best_model.pth"
    
    model = get_model(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️  Model weights not found at {model_path}, using random initialization")
    
    model.eval()
    return model

def evaluate_model():
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_img_dir = "data/val/Color_Images"
    val_mask_dir = "data/val/Segmentation"
    
    val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))
    
    if not val_imgs:
        print(f"Error: No validation images found in {val_img_dir}")
        return
    
    print(f"Evaluating FCN ResNet50 on {len(val_imgs)} validation images")
    
    model = load_model(NUM_CLASSES, DEVICE)
    
    val_ds = SegDataset(val_imgs, val_masks, augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
    
    total_dice = 0
    total_iou = 0
    per_class_dice_acc = torch.zeros(NUM_CLASSES).to(DEVICE)
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            outputs = model(imgs)
            
            d_mean, d_class = dice_score(outputs, masks, NUM_CLASSES)
            total_dice += d_mean.item()
            per_class_dice_acc += d_class
            
            iou = iou_score(outputs, masks, NUM_CLASSES)
            total_iou += iou.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_per_class_dice = per_class_dice_acc / len(val_loader)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results for FCN ResNet50")
    print(f"{'='*50}")
    print(f"Mean Dice Score: {avg_dice:.4f}")
    print(f"Mean IoU Score: {avg_iou:.4f}")
    print(f"\nPer-Class Dice Scores:")
    
    class_names = ["Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
                   "Flowers", "Logs", "Rocks", "Landscape", "Sky"]
    
    for i, (name, score) in enumerate(zip(class_names, avg_per_class_dice.cpu().numpy())):
        print(f"  {name:15s}: {score:.4f}")
    
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_CLASSES))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - FCN ResNet50')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix_fcn.png', dpi=300)
    plt.close()
    
    print(f"\nConfusion matrix saved to confusion_matrix_fcn.png")
    
    return {
        'model': 'fcn',
        'mean_dice': avg_dice,
        'mean_iou': avg_iou,
        'per_class_dice': avg_per_class_dice.cpu().numpy().tolist(),
        'confusion_matrix': cm.tolist()
    }

if __name__ == "__main__":
    evaluate_model()