import torch
import os
import glob
from torch.utils.data import DataLoader
from models.unet import get_model
from losses.losses import CombinedLoss
from dataset.dataset import SegDataset
from tqdm import tqdm
import numpy as np
from metrics import dice_score, iou_score
from early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.backends.cudnn.benchmark = True

def train():
    NUM_CLASSES = 10
    
    EPOCHS = 40
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_img_dir = "data/train/Color_Images"
    train_mask_dir = "data/train/Segmentation"
    val_img_dir = "data/val/Color_Images"
    val_mask_dir = "data/val/Segmentation"

    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.png")))
    train_masks = sorted(glob.glob(os.path.join(train_mask_dir, "*.png")))
    val_imgs = sorted(glob.glob(os.path.join(val_img_dir, "*.png")))
    val_masks = sorted(glob.glob(os.path.join(val_mask_dir, "*.png")))

    if not train_imgs:
        print(f"Warning: No training images found in {train_img_dir}.")
        print(f"Please ensure your images are in {train_img_dir} and masks in {train_mask_dir}")
        return

    model = get_model(NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    early_stopping = EarlyStopping(
        patience=8,
        path="checkpoints/best_model_unet.pth"
    )

    class_weights = torch.tensor([
        50.4137, 50.4162, 2.9984, 50.4182, 50.418, 50.4183, 50.4183, 50.418, 16.598, 3.2536
    ]).to(DEVICE)
    criterion = CombinedLoss(num_classes=NUM_CLASSES, weight=class_weights)

    train_ds = SegDataset(train_imgs, train_masks, augment=True)
    val_ds = SegDataset(val_imgs, val_masks, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        if epoch < 5:
            for param in model.encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.encoder.parameters():
                param.requires_grad = True
        
        model.train()
        epoch_loss = 0
        train_dice = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            
            dice, _ = dice_score(outputs, masks, NUM_CLASSES)
            train_dice += dice.item()
            pbar.set_postfix({"loss": loss.item(), "dice": dice.item()})

        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Dice: {avg_train_dice:.4f}"
        )

        model.eval()
        val_loss = 0
        total_dice = 0
        total_iou = 0
        per_class_dice_acc = torch.zeros(NUM_CLASSES).to(DEVICE)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                d_mean, d_class = dice_score(outputs, masks, NUM_CLASSES)
                total_dice += d_mean.item()
                per_class_dice_acc += d_class
                
                total_iou += iou_score(outputs, masks, NUM_CLASSES).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = total_dice / len(val_loader)
        avg_val_iou = total_iou / len(val_loader)
        avg_per_class_dice = per_class_dice_acc / len(val_loader)

        print(
            f"Epoch {epoch+1} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | "
            f"Val IoU: {avg_val_iou:.4f}"
        )
        print(f"Class Weights Used: {class_weights}")
        print(f"Per-Class Dice: {avg_per_class_dice.cpu().numpy()}")
        
        scheduler.step(avg_val_dice)
        early_stopping(avg_val_dice, model)

        if early_stopping.early_stop:
            print("🛑 Early stopping triggered. Training stopped.")
            break

        torch.save(model.state_dict(), f"checkpoints/last_model_unet.pth")

if __name__ == "__main__":
    train()