import torch
import os
import glob
from torch.utils.data import DataLoader
from models.deeplabv3plus import get_model
from dataset.dataset import SegDataset
from metrics import dice_score, iou_score
from tqdm import tqdm

def evaluate():
    NUM_CLASSES = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "checkpoints/best_model.pth"
    
    model = get_model(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    val_img_dir = "data/val/Color_Images"
    val_mask_dir = "data/val/Segmentation"
    val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))
    
    val_ds = SegDataset(val_imgs, val_masks, augment=False)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    total_dice = 0
    total_iou = 0
    per_class_dice_acc = torch.zeros(NUM_CLASSES).to(DEVICE)
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            
            d_mean, d_class = dice_score(outputs, masks, NUM_CLASSES)
            total_dice += d_mean.item()
            per_class_dice_acc += d_class
            total_iou += iou_score(outputs, masks, NUM_CLASSES).item()
            
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_per_class = per_class_dice_acc / len(val_loader)
    
    print(f"\n--- Final Results ---")
    print(f"Mean Dice: {avg_dice:.4f}")
    print(f"Mean IoU: {avg_iou:.4f}")
    
    class_names = [
        "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
        "Flowers", "Logs", "Rocks", "Landscape", "Sky"
    ]
    print("\nPer-Class Dice:")
    for i, name in enumerate(class_names):
        print(f"{name}: {avg_per_class[i]:.4f}")

if __name__ == "__main__":
    evaluate()
