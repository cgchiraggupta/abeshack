import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')

from dataset.dataset import OffRoadDataset
from models.attention_unet import get_model
from losses.losses import CombinedLoss
from metrics import compute_metrics
from early_stopping import EarlyStopping

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            dice, iou = compute_metrics(preds, masks)
            running_dice += dice
            running_iou += iou
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}',
                'IoU': f'{iou:.4f}'
            })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Dice/train', avg_dice, epoch)
    writer.add_scalar('IoU/train', avg_iou, epoch)
    
    return avg_loss, avg_dice, avg_iou

def validate_epoch(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            dice, iou = compute_metrics(preds, masks)
            running_dice += dice
            running_iou += iou
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}',
                'IoU': f'{iou:.4f}'
            })
    
    avg_loss = running_loss / len(dataloader)
    avg_dice = running_dice / len(dataloader)
    avg_iou = running_iou / len(dataloader)
    
    writer.add_scalar('Loss/val', avg_loss, epoch)
    writer.add_scalar('Dice/val', avg_dice, epoch)
    writer.add_scalar('IoU/val', avg_iou, epoch)
    
    return avg_loss, avg_dice, avg_iou

def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = OffRoadDataset(
        image_dir=config['data']['train_images'],
        mask_dir=config['data']['train_masks'],
        transform=True
    )
    
    val_dataset = OffRoadDataset(
        image_dir=config['data']['val_images'],
        mask_dir=config['data']['val_masks'],
        transform=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    model = get_model(num_classes=config['model']['num_classes'])
    model = model.to(device)
    
    criterion = CombinedLoss(
        num_classes=config['model']['num_classes'],
        weights=config['training']['class_weights']
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        verbose=True,
        delta=0.001
    )
    
    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    best_val_dice = 0.0
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device, epoch, writer
        )
        
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        scheduler.step(val_dice)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'val_loss': val_loss,
            }, os.path.join(config['training']['checkpoint_dir'], 'best_model.pth'))
            print(f"Saved best model with Val Dice: {val_dice:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou,
            'val_loss': val_loss,
        }, os.path.join(config['training']['checkpoint_dir'], 'last_model.pth'))
        
        early_stopping(val_dice)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    writer.close()
    print(f"\nTraining completed. Best Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Attention UNet for Off-Road Terrain Segmentation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)