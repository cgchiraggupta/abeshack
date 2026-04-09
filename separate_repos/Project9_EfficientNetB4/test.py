import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from models.efficientnet import EfficientNetSegmentation
from dataset.dataset import OffRoadDataset
from metrics import dice_score, iou_score

def get_test_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def test_model(model, dataloader, device):
    model.eval()
    all_dice_scores = []
    all_iou_scores = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            dice = dice_score(preds, masks, num_classes=11)
            iou = iou_score(preds, masks, num_classes=11)
            
            all_dice_scores.append(dice.cpu().numpy())
            all_iou_scores.append(iou.cpu().numpy())
            
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())
            
            pbar.set_postfix({'dice': dice.mean().item(), 'iou': iou.mean().item()})
    
    all_dice_scores = np.concatenate(all_dice_scores)
    all_iou_scores = np.concatenate(all_iou_scores)
    
    return all_dice_scores, all_iou_scores, all_predictions, all_targets

def plot_confusion_matrix(predictions, targets, class_names):
    cm = confusion_matrix(targets, predictions, labels=range(len(class_names)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Confusion matrix saved as confusion_matrix.png")

def plot_class_metrics(dice_scores, iou_scores, class_names):
    mean_dice = np.mean(dice_scores, axis=0)
    mean_iou = np.mean(iou_scores, axis=0)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, mean_dice, width, label='Dice Score', color='skyblue')
    rects2 = ax.bar(x + width/2, mean_iou, width, label='IoU Score', color='lightcoral')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Scores')
    ax.set_title('Per-Class Dice and IoU Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Class metrics plot saved as class_metrics.png")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_dataset = OffRoadDataset(
        image_dir='data/testImages/Color_Images',
        mask_dir='data/testImages/Segmentation',
        transform=get_test_transforms()
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    
    model = EfficientNetSegmentation(num_classes=11, variant='b4', pretrained=False).to(device)
    model.load_state_dict(torch.load('efficientnetb4_final.pth', map_location=device))
    print("Model loaded successfully")
    
    dice_scores, iou_scores, predictions, targets = test_model(model, test_loader, device)
    
    class_names = [
        'Background', 'Trees', 'Lush Bushes', 'Dry Bushes', 'Grass',
        'Concrete', 'Rocks', 'Water', 'Dirt', 'Mud', 'Snow'
    ]
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    print(f"\nOverall Metrics:")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"Mean IoU Score: {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    
    print(f"\nPer-Class Dice Scores:")
    for i, name in enumerate(class_names):
        print(f"{name:15s}: {np.mean(dice_scores[:, i]):.4f} ± {np.std(dice_scores[:, i]):.4f}")
    
    print(f"\nPer-Class IoU Scores:")
    for i, name in enumerate(class_names):
        print(f"{name:15s}: {np.mean(iou_scores[:, i]):.4f} ± {np.std(iou_scores[:, i]):.4f}")
    
    plot_confusion_matrix(predictions, targets, class_names)
    plot_class_metrics(dice_scores, iou_scores, class_names)
    
    print("\nDetailed Classification Report:")
    print(classification_report(targets, predictions, target_names=class_names, digits=4))
    
    results = {
        'mean_dice': float(np.mean(dice_scores)),
        'std_dice': float(np.std(dice_scores)),
        'mean_iou': float(np.mean(iou_scores)),
        'std_iou': float(np.std(iou_scores)),
        'per_class_dice': {name: float(np.mean(dice_scores[:, i])) for i, name in enumerate(class_names)},
        'per_class_iou': {name: float(np.mean(iou_scores[:, i])) for i, name in enumerate(class_names)}
    }
    
    import json
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nDetailed results saved to test_results.json")

if __name__ == '__main__':
    main()