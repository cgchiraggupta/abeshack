import torch
import numpy as np

def dice_coefficient(pred, target, smooth=1e-8):
    """
    Compute Dice coefficient between prediction and target masks.
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        pred_flat = pred[i].flatten()
        target_flat = target[i].flatten()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    return np.mean(dice_scores)

def iou_score(pred, target, smooth=1e-8):
    """
    Compute Intersection over Union (IoU) between prediction and target masks.
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        IoU score
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    batch_size = pred.shape[0]
    iou_scores = []
    
    for i in range(batch_size):
        pred_flat = pred[i].flatten()
        target_flat = target[i].flatten()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    return np.mean(iou_scores)

def pixel_accuracy(pred, target):
    """
    Compute pixel accuracy between prediction and target masks.
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
    
    Returns:
        Pixel accuracy
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    batch_size = pred.shape[0]
    acc_scores = []
    
    for i in range(batch_size):
        correct = (pred[i] == target[i]).sum().float()
        total = target[i].numel()
        
        accuracy = correct / total
        acc_scores.append(accuracy.item())
    
    return np.mean(acc_scores)

def compute_metrics(pred, target):
    """
    Compute multiple metrics for semantic segmentation.
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
    
    Returns:
        Dictionary containing Dice, IoU, and pixel accuracy
    """
    dice = dice_coefficient(pred, target)
    iou = iou_score(pred, target)
    accuracy = pixel_accuracy(pred, target)
    
    return dice, iou

def per_class_metrics(pred, target, num_classes):
    """
    Compute per-class Dice and IoU scores.
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
        num_classes: Number of classes
    
    Returns:
        Dictionary with per-class metrics
    """
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    batch_size = pred.shape[0]
    class_dice = {i: [] for i in range(num_classes)}
    class_iou = {i: [] for i in range(num_classes)}
    
    for i in range(batch_size):
        for class_idx in range(num_classes):
            pred_class = (pred[i] == class_idx)
            target_class = (target[i] == class_idx)
            
            if target_class.sum() > 0 or pred_class.sum() > 0:
                intersection = (pred_class & target_class).sum().float()
                union = (pred_class | target_class).sum().float()
                
                if union > 0:
                    iou_score = intersection / union
                    class_iou[class_idx].append(iou_score.item())
                
                if target_class.sum() > 0 or pred_class.sum() > 0:
                    dice_score = 2 * intersection / (pred_class.sum() + target_class.sum() + 1e-8)
                    class_dice[class_idx].append(dice_score.item())
    
    avg_class_dice = {k: np.mean(v) if v else 0 for k, v in class_dice.items()}
    avg_class_iou = {k: np.mean(v) if v else 0 for k, v in class_iou.items()}
    
    return {
        'per_class_dice': avg_class_dice,
        'per_class_iou': avg_class_iou,
        'mean_dice': np.mean(list(avg_class_dice.values())),
        'mean_iou': np.mean(list(avg_class_iou.values()))
    }

def confusion_matrix_metrics(pred, target, num_classes):
    """
    Compute confusion matrix and related metrics.
    
    Args:
        pred: Prediction tensor (B, H, W) or (H, W)
        target: Target tensor (B, H, W) or (H, W)
        num_classes: Number of classes
    
    Returns:
        Confusion matrix and metrics
    """
    if len(pred.shape) == 2:
        pred = pred.flatten()
        target = target.flatten()
    else:
        pred = pred.view(-1)
        target = target.view(-1)
    
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for t, p in zip(target, pred):
        cm[t.long(), p.long()] += 1
    
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1 = torch.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        if tp + fp > 0:
            precision[i] = tp / (tp + fp)
        if tp + fn > 0:
            recall[i] = tp / (tp + fn)
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    
    return {
        'confusion_matrix': cm.numpy(),
        'precision': precision.numpy(),
        'recall': recall.numpy(),
        'f1': f1.numpy(),
        'overall_accuracy': cm.diag().sum() / cm.sum()
    }

if __name__ == "__main__":
    # Test the metrics functions
    pred = torch.randint(0, 10, (2, 256, 256))
    target = torch.randint(0, 10, (2, 256, 256))
    
    dice, iou = compute_metrics(pred, target)
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    
    per_class = per_class_metrics(pred, target, 10)
    print(f"\nPer-class Dice: {per_class['per_class_dice']}")
    print(f"Mean Dice: {per_class['mean_dice']:.4f}")
    print(f"Mean IoU: {per_class['mean_iou']:.4f}")
    
    cm_metrics = confusion_matrix_metrics(pred, target, 10)
    print(f"\nOverall Accuracy: {cm_metrics['overall_accuracy']:.4f}")