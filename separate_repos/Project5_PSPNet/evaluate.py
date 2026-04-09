import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models.pspnet import PSPNet
from dataset.dataset import OffRoadDataset
from metrics import dice_score, iou_score

def get_eval_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def evaluate_model(model, dataloader, device):
    model.eval()
    all_dice_scores = []
    all_iou_scores = []
    all_precisions = []
    all_recalls = []
    all_f1_scores = []
    all_accuracies = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            dice = dice_score(preds, masks, num_classes=11)
            iou = iou_score(preds, masks, num_classes=11)
            
            preds_flat = preds.cpu().numpy().flatten()
            masks_flat = masks.cpu().numpy().flatten()
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                masks_flat, preds_flat, average='weighted', zero_division=0
            )
            accuracy = accuracy_score(masks_flat, preds_flat)
            
            all_dice_scores.append(dice.cpu().numpy())
            all_iou_scores.append(iou.cpu().numpy())
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1_scores.append(f1)
            all_accuracies.append(accuracy)
            
            all_predictions.extend(preds_flat)
            all_targets.extend(masks_flat)
            
            pbar.set_postfix({
                'dice': dice.mean().item(),
                'iou': iou.mean().item(),
                'f1': f1
            })
    
    all_dice_scores = np.concatenate(all_dice_scores)
    all_iou_scores = np.concatenate(all_iou_scores)
    
    return {
        'dice_scores': all_dice_scores,
        'iou_scores': all_iou_scores,
        'precision': np.mean(all_precisions),
        'recall': np.mean(all_recalls),
        'f1_score': np.mean(all_f1_scores),
        'accuracy': np.mean(all_accuracies),
        'predictions': all_predictions,
        'targets': all_targets
    }

def plot_metrics_comparison(results, class_names):
    mean_dice = np.mean(results['dice_scores'], axis=0)
    mean_iou = np.mean(results['iou_scores'], axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax1.bar(x - width/2, mean_dice, width, label='Dice Score', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, mean_iou, width, label='IoU Score', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Scores')
    ax1.set_title('Per-Class Segmentation Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, (dice, iou) in enumerate(zip(mean_dice, mean_iou)):
        ax1.text(i - width/2, dice + 0.01, f'{dice:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, iou + 0.01, f'{iou:.3f}', ha='center', va='bottom', fontsize=8)
    
    overall_metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    overall_values = [
        results['precision'],
        results['recall'],
        results['f1_score'],
        results['accuracy']
    ]
    
    colors = ['lightgreen', 'lightblue', 'gold', 'lightcoral']
    bars = ax2.bar(overall_metrics, overall_values, color=colors, alpha=0.8)
    ax2.set_ylabel('Score')
    ax2.set_title('Overall Classification Metrics')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, overall_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Evaluation metrics plot saved as evaluation_metrics.png")

def generate_detailed_report(results, class_names):
    report = {
        'model': 'PSPNet ResNet101',
        'overall_metrics': {
            'mean_dice': float(np.mean(results['dice_scores'])),
            'std_dice': float(np.std(results['dice_scores'])),
            'mean_iou': float(np.mean(results['iou_scores'])),
            'std_iou': float(np.std(results['iou_scores'])),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'accuracy': float(results['accuracy'])
        },
        'per_class_metrics': {}
    }
    
    for i, name in enumerate(class_names):
        report['per_class_metrics'][name] = {
            'dice_mean': float(np.mean(results['dice_scores'][:, i])),
            'dice_std': float(np.std(results['dice_scores'][:, i])),
            'iou_mean': float(np.mean(results['iou_scores'][:, i])),
            'iou_std': float(np.std(results['iou_scores'][:, i]))
        }
    
    with open('evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("Detailed evaluation report saved as evaluation_report.json")
    
    return report

def print_summary(report, class_names):
    print("\n" + "="*60)
    print("EVALUATION SUMMARY - PSPNet ResNet101")
    print("="*60)
    
    print("\nOVERALL METRICS:")
    print(f"{'Metric':<15} {'Value':<10} {'Std Dev':<10}")
    print("-" * 40)
    print(f"{'Dice Score':<15} {report['overall_metrics']['mean_dice']:.4f}      {report['overall_metrics']['std_dice']:.4f}")
    print(f"{'IoU Score':<15} {report['overall_metrics']['mean_iou']:.4f}      {report['overall_metrics']['std_iou']:.4f}")
    print(f"{'Precision':<15} {report['overall_metrics']['precision']:.4f}")
    print(f"{'Recall':<15} {report['overall_metrics']['recall']:.4f}")
    print(f"{'F1-Score':<15} {report['overall_metrics']['f1_score']:.4f}")
    print(f"{'Accuracy':<15} {report['overall_metrics']['accuracy']:.4f}")
    
    print(f"\nPER-CLASS DICE SCORES:")
    print(f"{'Class':<15} {'Mean':<10} {'Std Dev':<10}")
    print("-" * 40)
    for name in class_names:
        metrics = report['per_class_metrics'][name]
        print(f"{name:<15} {metrics['dice_mean']:.4f}      {metrics['dice_std']:.4f}")
    
    print(f"\nPER-CLASS IoU SCORES:")
    print(f"{'Class':<15} {'Mean':<10} {'Std Dev':<10}")
    print("-" * 40)
    for name in class_names:
        metrics = report['per_class_metrics'][name]
        print(f"{name:<15} {metrics['iou_mean']:.4f}      {metrics['iou_std']:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    eval_dataset = OffRoadDataset(
        image_dir='data/val/Color_Images',
        mask_dir='data/val/Segmentation',
        transform=get_eval_transforms()
    )
    
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    model = PSPNet(num_classes=11, backbone='resnet101', pretrained=False).to(device)
    model.load_state_dict(torch.load('pspnet_final.pth', map_location=device))
    print("Model loaded successfully")
    
    results = evaluate_model(model, eval_loader, device)
    
    class_names = [
        'Background', 'Trees', 'Lush Bushes', 'Dry Bushes', 'Grass',
        'Concrete', 'Rocks', 'Water', 'Dirt', 'Mud', 'Snow'
    ]
    
    report = generate_detailed_report(results, class_names)
    plot_metrics_comparison(results, class_names)
    print_summary(report, class_names)

if __name__ == '__main__':
    main()