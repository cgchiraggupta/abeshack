import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os
import time
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Custom imports
from dataset.dataset import OffRoadDataset, get_transforms, compute_class_weights
from models.deeplabv3_resnet50 import DeepLabV3ResNet50
from losses.losses import CombinedLoss
from metrics import dice_score, iou_score, SegmentationMetrics

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (images, masks, _) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            # Forward pass
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Learning rate scheduler step (if per iteration)
        if scheduler and hasattr(scheduler, 'step'):
            scheduler.step()
        
        # Calculate metrics
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            batch_dice = dice_score(preds, masks)
            batch_iou = iou_score(preds, masks)
        
        # Update totals
        total_loss += loss.item()
        total_dice += batch_dice.item()
        total_iou += batch_iou.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{batch_dice.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
    
    # Calculate averages
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    
    return avg_loss, avg_dice, avg_iou

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    
    # Initialize metrics calculator
    metrics_calc = SegmentationMetrics(num_classes=10, device=device)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        
        for images, masks, _ in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)
            
            # Calculate metrics
            preds = torch.argmax(outputs, dim=1)
            batch_dice = dice_score(preds, masks)
            batch_iou = iou_score(preds, masks)
            
            # Update metrics calculator
            metrics_calc.update(preds, masks)
            
            # Update totals
            total_loss += loss.item()
            total_dice += batch_dice.item()
            total_iou += batch_iou.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{batch_dice.item():.4f}'
            })
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    
    # Get detailed metrics
    metrics_dict = metrics_calc.compute()
    
    return avg_loss, avg_dice, avg_iou, metrics_dict

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best=False, checkpoint_dir='checkpoints'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'model_config': {
            'num_classes': 10,
            'backbone': 'resnet50',
            'model_type': 'deeplabv3'
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_path)
        print(f"Saved best model with Dice: {metrics['val_dice']:.4f}")
    
    # Save latest model
    latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
    torch.save(model.state_dict(), latest_path)
    
    return checkpoint_path

def train_model(config):
    """Main training function"""
    print("=" * 60)
    print("Off-Road Terrain Segmentation - DeepLabV3+ ResNet50")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Create datasets and dataloaders
    print("\nLoading datasets...")
    
    # Get transforms
    train_transform = get_transforms('train', config['image_size'])
    val_transform = get_transforms('val', config['image_size'])
    
    # Create datasets
    train_dataset = OffRoadDataset(
        data_dir=config['data_dir'],
        split='train',
        transform=train_transform,
        debug=config['debug']
    )
    
    val_dataset = OffRoadDataset(
        data_dir=config['data_dir'],
        split='val',
        transform=val_transform,
        debug=config['debug']
    )
    
    # Calculate class weights
    print("Computing class weights...")
    class_weights = compute_class_weights(config['data_dir'], split='train')
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights.tolist()}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of batches - Train: {len(train_loader)}, Val: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = DeepLabV3ResNet50(num_classes=config['num_classes'], pretrained=True)
    model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    print("\nCreating loss function...")
    criterion = CombinedLoss(class_weights=class_weights, device=device)
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    print("Creating scheduler...")
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config['scheduler_t0'],
        T_mult=config['scheduler_tmult'],
        eta_min=config['scheduler_eta_min']
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Training variables
    best_val_dice = 0
    best_val_iou = 0
    early_stop_counter = 0
    train_history = {
        'train_loss': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_dice': [], 'val_iou': [],
        'learning_rates': [], 'epoch_times': []
    }
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        print("-" * 40)
        
        start_time = time.time()
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, scheduler
        )
        
        # Validate
        val_loss, val_dice, val_iou, val_metrics = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Update history
        train_history['train_loss'].append(train_loss)
        train_history['train_dice'].append(train_dice)
        train_history['train_iou'].append(train_iou)
        train_history['val_loss'].append(val_loss)
        train_history['val_dice'].append(val_dice)
        train_history['val_iou'].append(val_iou)
        train_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        train_history['epoch_times'].append(epoch_time)
        
        # Print epoch results
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Check for best model
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            best_val_iou = val_iou
            early_stop_counter = 0
            print(f"🔥 New best model! Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        else:
            early_stop_counter += 1
            print(f"Early stop counter: {early_stop_counter}/{config['early_stop_patience']}")
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_dice': val_dice,
            'val_iou': val_iou,
            'epoch': epoch
        }
        
        checkpoint_path = save_checkpoint(
            model, optimizer, scheduler, epoch, metrics,
            is_best=is_best, checkpoint_dir=config['checkpoint_dir']
        )
        
        # Early stopping
        if early_stop_counter >= config['early_stop_patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        # Update scheduler (per epoch)
        if scheduler and not hasattr(scheduler, 'step'):
            scheduler.step()
    
    # Training complete
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Save training history
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Save final model
    final_model_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Print best results
    print(f"\nBest validation results:")
    print(f"Dice Score: {best_val_dice:.4f}")
    print(f"IoU Score: {best_val_iou:.4f}")
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss curves
        axes[0, 0].plot(train_history['train_loss'], label='Train')
        axes[0, 0].plot(train_history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Dice score curves
        axes[0, 1].plot(train_history['train_dice'], label='Train')
        axes[0, 1].plot(train_history['val_dice'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # IoU score curves
        axes[1, 0].plot(train_history['train_iou'], label='Train')
        axes[1, 0].plot(train_history['val_iou'], label='Val')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('Training and Validation IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning rate curve
        axes[1, 1].plot(train_history['learning_rates'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(config['checkpoint_dir'], 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_path}")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    return model, train_history, best_val_dice, best_val_iou

def main():
    """Main function with default configuration"""
    # Configuration
    config = {
        # Data
        'data_dir': 'data',
        'image_size': 512,
        'num_classes': 10,
        
        # Training
        'epochs': 40,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        
        # Scheduler
        'scheduler_t0': 10,
        'scheduler_tmult': 2,
        'scheduler_eta_min': 1e-6,
        
        # Early stopping
        'early_stop_patience': 10,
        
        # System
        'num_workers': 4,
        'checkpoint_dir': 'checkpoints',
        'debug': False,
        
        # Model
        'model_name': 'DeepLabV3+ ResNet50'
    }
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config['checkpoint_dir'], 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")
    
    # Train model
    model, history, best_dice, best_iou = train_model(config)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Best Validation IoU: {best_iou:.4f}")
    print(f"Total epochs trained: {len(history['train_loss'])}")
    print(f"Checkpoints saved in: {config['checkpoint_dir']}")
    print("=" * 60)

if __name__ == "__main__":
    main()