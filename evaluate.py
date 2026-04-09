import torch
import os
import glob
import argparse
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from dataset.dataset import SegDataset
from metrics import dice_score, iou_score

def load_model(model_name, num_classes=10, device="cuda"):
    if model_name == "deeplabv3plus":
        from models.deeplabv3plus import get_model
        model_path = "checkpoints/best_model_deeplabv3plus.pth"
    elif model_name == "unet":
        from models.unet import get_model
        model_path = "checkpoints/best_model_unet.pth"
    elif model_name == "pspnet":
        from models.pspnet import get_model
        model_path = "checkpoints/best_model_pspnet.pth"
    elif model_name == "fpn":
        from models.fpn import get_model
        model_path = "checkpoints/best_model_fpn.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = get_model(num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️  Model weights not found at {model_path}, using random initialization")
    
    model.eval()
    return model

def evaluate_model(model_name="deeplabv3plus"):
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_img_dir = "data/val/Color_Images"
    val_mask_dir = "data/val/Segmentation"
    
    val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))
    
    if not val_imgs:
        print(f"Error: No validation images found in {val_img_dir}")
        return
    
    print(f"Evaluating {model_name} on {len(val_imgs)} validation images")
    
    model = load_model(model_name, NUM_CLASSES, DEVICE)
    
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
    print(f"Evaluation Results for {model_name.upper()}")
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
    plt.title(f'Confusion Matrix - {model_name.upper()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png', dpi=300)
    plt.close()
    
    print(f"\nConfusion matrix saved to confusion_matrix_{model_name}.png")
    
    return {
        'model': model_name,
        'mean_dice': avg_dice,
        'mean_iou': avg_iou,
        'per_class_dice': avg_per_class_dice.cpu().numpy().tolist(),
        'confusion_matrix': cm.tolist()
    }

def compare_all_models():
    models = ["deeplabv3plus", "unet", "pspnet", "fpn"]
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*60}")
        try:
            result = evaluate_model(model_name)
            results[model_name] = result
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    if results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':20s} {'Dice':10s} {'IoU':10s}")
        print(f"{'-'*40}")
        
        for model_name, result in results.items():
            print(f"{model_name:20s} {result['mean_dice']:10.4f} {result['mean_iou']:10.4f}")
        
        best_model = max(results.items(), key=lambda x: x[1]['mean_dice'])
        print(f"\n🏆 Best Model: {best_model[0]} (Dice: {best_model[1]['mean_dice']:.4f})")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate segmentation models')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['deeplabv3plus', 'unet', 'pspnet', 'fpn', 'all'],
                       help='Model to evaluate (default: all)')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        compare_all_models()
    else:
        evaluate_model(args.model)