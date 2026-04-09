import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset.dataset import OffRoadDataset
from models.efficientnet_b4 import EfficientNetB4Segmentation
from metrics import dice_score, iou_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model_path, data_dir, device='cuda', batch_size=4):
    """
    Evaluate the trained model on validation set
    """
    # Load model
    model = EfficientNetB4Segmentation(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Validation transforms
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create dataset and dataloader
    val_dataset = OffRoadDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize metrics
    total_dice = 0
    total_iou = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            batch_dice = dice_score(preds, masks)
            batch_iou = iou_score(preds, masks)
            
            total_dice += batch_dice.item() * images.size(0)
            total_iou += batch_iou.item() * images.size(0)
            
            # Store for confusion matrix
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
    
    # Calculate average metrics
    avg_dice = total_dice / len(val_dataset)
    avg_iou = total_iou / len(val_dataset)
    
    print(f"\nEvaluation Results:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions, labels=range(10))
    
    # Calculate per-class metrics
    class_names = [
        'Trees', 'Lush Bushes', 'Grass', 'Dirt', 'Sand', 
        'Water', 'Rocks', 'Bushes', 'Mud', 'Background'
    ]
    
    print("\nPer-class IoU Scores:")
    for i in range(10):
        if i == 9:  # Background class
            continue
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        class_iou = tp / (tp + fp + fn + 1e-8)
        print(f"{class_names[i]}: {class_iou:.4f}")
    
    # Save results
    results = {
        'model': 'EfficientNet-B4',
        'dice_score': avg_dice,
        'iou_score': avg_iou,
        'per_class_iou': {}
    }
    
    for i in range(10):
        if i == 9:
            continue
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        class_iou = tp / (tp + fp + fn + 1e-8)
        results['per_class_iou'][class_names[i]] = class_iou
    
    # Save to JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - EfficientNet-B4')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot per-class IoU
    plt.figure(figsize=(12, 6))
    class_iou_values = []
    for i in range(10):
        if i == 9:
            continue
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        class_iou = tp / (tp + fp + fn + 1e-8)
        class_iou_values.append(class_iou)
    
    plt.bar(class_names[:-1], class_iou_values)
    plt.xlabel('Classes')
    plt.ylabel('IoU Score')
    plt.title('Per-class IoU Scores - EfficientNet-B4')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig('per_class_iou.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def visualize_predictions(model_path, data_dir, device='cuda', num_samples=5):
    """
    Visualize model predictions on sample images
    """
    # Load model
    model = EfficientNetB4Segmentation(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Validation transforms
    val_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create dataset
    val_dataset = OffRoadDataset(
        data_dir=data_dir,
        split='val',
        transform=val_transform
    )
    
    # Class colors for visualization
    class_colors = [
        (34, 139, 34),    # Trees - Forest Green
        (0, 100, 0),      # Lush Bushes - Dark Green
        (124, 252, 0),    # Grass - Lawn Green
        (139, 69, 19),    # Dirt - Saddle Brown
        (238, 203, 173),  # Sand - Burlywood
        (30, 144, 255),   # Water - Dodger Blue
        (128, 128, 128),  # Rocks - Gray
        (0, 128, 0),      # Bushes - Green
        (101, 67, 33),    # Mud - Dark Brown
        (0, 0, 0)         # Background - Black
    ]
    
    # Select random samples
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx in indices:
            image, mask = val_dataset[idx]
            image_tensor = image.unsqueeze(0).to(device)
            
            # Get prediction
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Convert tensor to numpy for visualization
            image_np = image.permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            image_np = image_np.astype(np.uint8)
            
            # Create colored masks
            gt_colored = np.zeros((512, 512, 3), dtype=np.uint8)
            pred_colored = np.zeros((512, 512, 3), dtype=np.uint8)
            
            for class_idx in range(10):
                gt_colored[mask == class_idx] = class_colors[class_idx]
                pred_colored[pred == class_idx] = class_colors[class_idx]
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image_np)
            axes[0].set_title('Input Image')
            axes[0].axis('off')
            
            axes[1].imshow(gt_colored)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(pred_colored)
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.suptitle(f'Sample {idx+1} - EfficientNet-B4', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'prediction_sample_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Saved {num_samples} prediction visualizations")

if __name__ == "__main__":
    # Example usage
    model_path = "best_model.pth"
    data_dir = "data"
    
    # Evaluate model
    results = evaluate_model(model_path, data_dir, device='cuda')
    
    # Visualize predictions
    visualize_predictions(model_path, data_dir, device='cuda', num_samples=5)