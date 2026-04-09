import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('.')

from dataset.dataset import OffRoadDataset
from models.attention_unet import get_model
from metrics import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def evaluate_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_dataset = OffRoadDataset(
        image_dir=config['data']['val_images'],
        mask_dir=config['data']['val_masks'],
        transform=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers'],
        pin_memory=True
    )
    
    model = get_model(num_classes=config['model']['num_classes'])
    model = model.to(device)
    
    checkpoint_path = config['evaluation']['checkpoint_path']
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Checkpoint metrics - Dice: {checkpoint.get('val_dice', 'N/A'):.4f}, "
              f"IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
        return
    
    model.eval()
    
    all_dice_scores = []
    all_iou_scores = []
    all_predictions = []
    all_targets = []
    
    class_dice_scores = {i: [] for i in range(config['model']['num_classes'])}
    class_iou_scores = {i: [] for i in range(config['model']['num_classes'])}
    
    os.makedirs(config['evaluation']['output_dir'], exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Evaluation')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            dice, iou = compute_metrics(preds, masks)
            all_dice_scores.append(dice)
            all_iou_scores.append(iou)
            
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            
            for class_idx in range(config['model']['num_classes']):
                class_mask = (masks == class_idx)
                class_pred = (preds == class_idx)
                
                if class_mask.sum() > 0 or class_pred.sum() > 0:
                    intersection = (class_pred & class_mask).sum().float()
                    union = (class_pred | class_mask).sum().float()
                    
                    if union > 0:
                        class_iou = intersection / union
                        class_iou_scores[class_idx].append(class_iou.item())
                    
                    if class_mask.sum() > 0 or class_pred.sum() > 0:
                        dice_score = 2 * intersection / (class_pred.sum() + class_mask.sum() + 1e-8)
                        class_dice_scores[class_idx].append(dice_score.item())
            
            if batch_idx < config['evaluation']['num_visualizations']:
                for i in range(min(images.size(0), 2)):
                    idx = batch_idx * images.size(0) + i
                    
                    image_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    image_np = (image_np * 255).astype(np.uint8)
                    
                    mask_np = masks[i].cpu().numpy()
                    pred_np = preds[i].cpu().numpy()
                    
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(image_np)
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(mask_np, cmap='tab20', vmin=0, vmax=config['model']['num_classes']-1)
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(pred_np, cmap='tab20', vmin=0, vmax=config['model']['num_classes']-1)
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(config['evaluation']['output_dir'], f'eval_{idx}.png'), dpi=150, bbox_inches='tight')
                    plt.close()
    
    avg_dice = np.mean(all_dice_scores)
    avg_iou = np.mean(all_iou_scores)
    
    print(f"\nEvaluation Results:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    
    print(f"\nPer-class Dice Scores:")
    for class_idx in range(config['model']['num_classes']):
        if class_dice_scores[class_idx]:
            class_avg_dice = np.mean(class_dice_scores[class_idx])
            print(f"  Class {class_idx}: {class_avg_dice:.4f}")
    
    print(f"\nPer-class IoU Scores:")
    for class_idx in range(config['model']['num_classes']):
        if class_iou_scores[class_idx]:
            class_avg_iou = np.mean(class_iou_scores[class_idx])
            print(f"  Class {class_idx}: {class_avg_iou:.4f}")
    
    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(config['model']['num_classes'])))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(config['model']['num_classes'])],
                yticklabels=[f'Class {i}' for i in range(config['model']['num_classes'])])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(config['evaluation']['output_dir'], 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    class_names = [f'Class_{i}' for i in range(config['model']['num_classes'])]
    report = classification_report(all_targets, all_predictions, 
                                   target_names=class_names, 
                                   output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(config['evaluation']['output_dir'], 'classification_report.csv'))
    
    metrics_df = pd.DataFrame({
        'Class': list(range(config['model']['num_classes'])),
        'Dice': [np.mean(class_dice_scores[i]) if class_dice_scores[i] else 0 for i in range(config['model']['num_classes'])],
        'IoU': [np.mean(class_iou_scores[i]) if class_iou_scores[i] else 0 for i in range(config['model']['num_classes'])]
    })
    metrics_df.to_csv(os.path.join(config['evaluation']['output_dir'], 'per_class_metrics.csv'), index=False)
    
    results = {
        'avg_dice': avg_dice,
        'avg_iou': avg_iou,
        'class_dice': {str(k): np.mean(v) if v else 0 for k, v in class_dice_scores.items()},
        'class_iou': {str(k): np.mean(v) if v else 0 for k, v in class_iou_scores.items()},
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    import json
    with open(os.path.join(config['evaluation']['output_dir'], 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nDetailed evaluation results saved to {config['evaluation']['output_dir']}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Attention UNet for Off-Road Terrain Segmentation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate_model(config)