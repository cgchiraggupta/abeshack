import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.pspnet import PSPNet
from dataset.dataset import OffRoadDataset
from losses.losses import CombinedLoss
from metrics import dice_score, iou_score

def compute_class_weights():
    return torch.tensor([
        1.0,    # 0: Background
        2.5,    # 100: Trees
        3.0,    # 200: Lush Bushes
        2.8,    # 300: Dry Bushes
        2.2,    # 400: Grass
        3.5,    # 500: Concrete
        3.2,    # 600: Rocks
        2.7,    # 700: Water
        3.8,    # 800: Dirt
        4.0,    # 900: Mud
        3.3     # 1000: Snow
    ]).cuda()

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(512, 512, scale=(0.5, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)
        ], p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5)
        ], p=0.5),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5)
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    dice_scores = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            dice = dice_score(preds, masks, num_classes=11)
            dice_scores.append(dice.mean().item())
        
        pbar.set_postfix({'loss': loss.item(), 'dice': dice.mean().item()})
    
    return total_loss / len(dataloader), np.mean(dice_scores)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            dice = dice_score(preds, masks, num_classes=11)
            iou = iou_score(preds, masks, num_classes=11)
            
            dice_scores.append(dice.mean().item())
            iou_scores.append(iou.mean().item())
            
            pbar.set_postfix({'val_loss': loss.item(), 'val_dice': dice.mean().item()})
    
    return total_loss / len(dataloader), np.mean(dice_scores), np.mean(iou_scores)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = OffRoadDataset(
        image_dir='data/train/Color_Images',
        mask_dir='data/train/Segmentation',
        transform=get_train_transforms()
    )
    
    val_dataset = OffRoadDataset(
        image_dir='data/val/Color_Images',
        mask_dir='data/val/Segmentation',
        transform=get_val_transforms()
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    model = PSPNet(num_classes=11, backbone='resnet101', pretrained=True).to(device)
    
    class_weights = compute_class_weights()
    criterion = CombinedLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    scaler = GradScaler()
    
    num_epochs = 40
    best_dice = 0
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step(val_dice)
        
        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_iou': val_iou
            }, 'best_model.pth')
            print(f"Saved best model with Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    print(f"\nTraining completed. Best Val Dice: {best_dice:.4f}")
    
    final_model = PSPNet(num_classes=11, backbone='resnet101', pretrained=False).to(device)
    checkpoint = torch.load('best_model.pth', map_location=device)
    final_model.load_state_dict(checkpoint['model_state_dict'])
    torch.save(final_model.state_dict(), 'pspnet_final.pth')
    print("Final model saved as pspnet_final.pth")

if __name__ == '__main__':
    main()