import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient (F1 score) for segmentation
    
    Args:
        pred: Predicted segmentation mask (B, H, W) or (H, W)
        target: Ground truth mask (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        dice: Dice coefficient
    """
    # Ensure tensors are on same device and have same shape
    if isinstance(pred, torch.Tensor):
        pred = pred.to(target.device)
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Calculate Dice coefficient
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return dice

def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU/Jaccard index)
    
    Args:
        pred: Predicted segmentation mask (B, H, W) or (H, W)
        target: Ground truth mask (B, H, W) or (H, W)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        iou: IoU score
    """
    # Ensure tensors are on same device and have same shape
    if isinstance(pred, torch.Tensor):
        pred = pred.to(target.device)
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate intersection and union
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # Calculate IoU
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def precision_score(pred, target, smooth=1e-6):
    """
    Calculate precision (positive predictive value)
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        precision: Precision score
    """
    # Ensure tensors are on same device
    if isinstance(pred, torch.Tensor):
        pred = pred.to(target.device)
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate true positives and predicted positives
    true_positives = (pred_flat * target_flat).sum()
    predicted_positives = pred_flat.sum()
    
    # Calculate precision
    precision = (true_positives + smooth) / (predicted_positives + smooth)
    
    return precision

def recall_score(pred, target, smooth=1e-6):
    """
    Calculate recall (sensitivity/true positive rate)
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        recall: Recall score
    """
    # Ensure tensors are on same device
    if isinstance(pred, torch.Tensor):
        pred = pred.to(target.device)
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate true positives and actual positives
    true_positives = (pred_flat * target_flat).sum()
    actual_positives = target_flat.sum()
    
    # Calculate recall
    recall = (true_positives + smooth) / (actual_positives + smooth)
    
    return recall

def f1_score(pred, target, smooth=1e-6):
    """
    Calculate F1 score (harmonic mean of precision and recall)
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        smooth: Smoothing factor
    
    Returns:
        f1: F1 score
    """
    precision = precision_score(pred, target, smooth)
    recall = recall_score(pred, target, smooth)
    
    # Calculate F1 score
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    
    return f1

def accuracy_score(pred, target):
    """
    Calculate accuracy
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
    
    Returns:
        accuracy: Accuracy score
    """
    # Ensure tensors are on same device
    if isinstance(pred, torch.Tensor):
        pred = pred.to(target.device)
    
    # Flatten tensors
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate accuracy
    correct = (pred_flat == target_flat).sum()
    total = pred_flat.numel()
    
    accuracy = correct.float() / total
    
    return accuracy

def per_class_iou(pred, target, num_classes=10, smooth=1e-6):
    """
    Calculate IoU for each class
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        iou_per_class: IoU scores for each class
    """
    iou_per_class = torch.zeros(num_classes, device=target.device)
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        # Calculate IoU for this class
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class[class_idx] = iou
    
    return iou_per_class

def per_class_dice(pred, target, num_classes=10, smooth=1e-6):
    """
    Calculate Dice coefficient for each class
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        dice_per_class: Dice scores for each class
    """
    dice_per_class = torch.zeros(num_classes, device=target.device)
    
    for class_idx in range(num_classes):
        # Create binary masks for current class
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        # Calculate Dice for this class
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class[class_idx] = dice
    
    return dice_per_class

def mean_iou(pred, target, num_classes=10, smooth=1e-6):
    """
    Calculate mean IoU across all classes
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        miou: Mean IoU score
    """
    iou_per_class = per_class_iou(pred, target, num_classes, smooth)
    
    # Exclude background class (class 9) from mean calculation
    valid_classes = iou_per_class[:9]  # Classes 0-8
    miou = valid_classes.mean()
    
    return miou

def mean_dice(pred, target, num_classes=10, smooth=1e-6):
    """
    Calculate mean Dice coefficient across all classes
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        mdice: Mean Dice score
    """
    dice_per_class = per_class_dice(pred, target, num_classes, smooth)
    
    # Exclude background class (class 9) from mean calculation
    valid_classes = dice_per_class[:9]  # Classes 0-8
    mdice = valid_classes.mean()
    
    return mdice

def compute_confusion_matrix(pred, target, num_classes=10):
    """
    Compute confusion matrix
    
    Args:
        pred: Predicted segmentation mask
        target: Ground truth mask
        num_classes: Number of classes
    
    Returns:
        cm: Confusion matrix (num_classes x num_classes)
    """
    # Convert to numpy if tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # Flatten arrays
    pred_flat = pred.ravel()
    target_flat = target.ravel()
    
    # Compute confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
    
    return cm

def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compute_class_weights(masks, num_classes=10, method='inverse_frequency'):
    """
    Compute class weights for imbalanced datasets
    
    Args:
        masks: List of ground truth masks
        num_classes: Number of classes
        method: Weight computation method ('inverse_frequency' or 'median_frequency')
    
    Returns:
        class_weights: Tensor of class weights
    """
    # Count pixel frequencies for each class
    class_counts = torch.zeros(num_classes)
    
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Flatten mask and count class occurrences
        flat_mask = mask_np.ravel()
        for class_idx in range(num_classes):
            class_counts[class_idx] += (flat_mask == class_idx).sum()
    
    if method == 'inverse_frequency':
        # Inverse of class frequencies
        total_pixels = class_counts.sum()
        class_weights = total_pixels / (num_classes * class_counts + 1e-8)
    
    elif method == 'median_frequency':
        # Median frequency balancing
        class_frequencies = class_counts / class_counts.sum()
        median_frequency = torch.median(class_frequencies)
        class_weights = median_frequency / (class_frequencies + 1e-8)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum()
    
    return class_weights

class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator
    """
    def __init__(self, num_classes=10, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        
        # Initialize accumulators
        self.reset()
    
    def reset(self):
        """Reset all accumulators"""
        self.total_dice = 0
        self.total_iou = 0
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0
        self.total_accuracy = 0
        self.total_samples = 0
        
        # Per-class accumulators
        self.class_iou = torch.zeros(self.num_classes, device=self.device)
        self.class_dice = torch.zeros(self.num_classes, device=self.device)
        self.class_counts = torch.zeros(self.num_classes, device=self.device)
    
    def update(self, pred, target):
        """
        Update metrics with new batch
        
        Args:
            pred: Predicted masks (B, H, W)
            target: Ground truth masks (B, H, W)
        """
        batch_size = pred.shape[0]
        
        for i in range(batch_size):
            pred_i = pred[i]
            target_i = target[i]
            
            # Calculate metrics
            dice = dice_score(pred_i, target_i)
            iou = iou_score(pred_i, target_i)
            precision = precision_score(pred_i, target_i)
            recall = recall_score(pred_i, target_i)
            f1 = f1_score(pred_i, target_i)
            accuracy = accuracy_score(pred_i, target_i)
            
            # Update totals
            self.total_dice += dice.item()
            self.total_iou += iou.item()
            self.total_precision += precision.item()
            self.total_recall += recall.item()
            self.total_f1 += f1.item()
            self.total_accuracy += accuracy.item()
            
            # Update per-class metrics
            class_iou_batch = per_class_iou(pred_i, target_i, self.num_classes)
            class_dice_batch = per_class_dice(pred_i, target_i, self.num_classes)
            
            self.class_iou += class_iou_batch
            self.class_dice += class_dice_batch
            self.class_counts += 1
        
        self.total_samples += batch_size
    
    def compute(self):
        """
        Compute final metrics
        
        Returns:
            metrics_dict: Dictionary containing all computed metrics
        """
        if self.total_samples == 0:
            return {}
        
        # Compute averages
        metrics = {
            'dice': self.total_dice / self.total_samples,
            'iou': self.total_iou / self.total_samples,
            'precision': self.total_precision / self.total_samples,
            'recall': self.total_recall / self.total_samples,
            'f1': self.total_f1 / self.total_samples,
            'accuracy': self.total_accuracy / self.total_samples,
            'mean_iou': self.class_iou[:9].mean().item() / max(self.class_counts[:9].max().item(), 1),
            'mean_dice': self.class_dice[:9].mean().item() / max(self.class_counts[:9].max().item(), 1)
        }
        
        # Add per-class metrics
        for class_idx in range(self.num_classes):
            if self.class_counts[class_idx] > 0:
                metrics[f'class_{class_idx}_iou'] = self.class_iou[class_idx].item() / self.class_counts[class_idx]
                metrics[f'class_{class_idx}_dice'] = self.class_dice[class_idx].item() / self.class_counts[class_idx]
        
        return metrics
    
    def get_summary(self, class_names=None):
        """
        Get formatted summary of metrics
        
        Args:
            class_names: List of class names (optional)
        
        Returns:
            summary_str: Formatted summary string
        """
        metrics = self.compute()
        
        if not metrics:
            return "No metrics computed yet."
        
        summary = "Segmentation Metrics Summary:\n"
        summary += "=" * 50 + "\n"
        
        # Overall metrics
        summary += "Overall Metrics:\n"
        summary += f"  Dice Score: {metrics['dice']:.4f}\n"
        summary += f"  IoU Score: {metrics['iou']:.4f}\n"
        summary += f"  Precision: {metrics['precision']:.4f}\n"
        summary += f"  Recall: {metrics['recall']:.4f}\n"
        summary += f"  F1 Score: {metrics['f1']:.4f}\n"
        summary += f"  Accuracy: {metrics['accuracy']:.4f}\n"
        summary += f"  Mean IoU: {metrics['mean_iou']:.4f}\n"
        summary += f"  Mean Dice: {metrics['mean_dice']:.4f}\n"
        
        # Per-class metrics
        if class_names and len(class_names) == self.num_classes:
            summary += "\nPer-class IoU Scores:\n"
            for class_idx in range(self.num_classes):
                if class_idx == 9:  # Skip background
                    continue
                iou_key = f'class_{class_idx}_iou'
                if iou_key in metrics:
                    summary += f"  {class_names[class_idx]}: {metrics[iou_key]:.4f}\n"
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample data
    batch_size = 4
    height, width = 512, 512
    num_classes = 10
    
    # Random predictions and targets
    pred = torch.randint(0, num_classes, (batch_size, height, width))
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Initialize metrics calculator
    metrics_calc = SegmentationMetrics(num_classes=num_classes)
    
    # Update with batch
    metrics_calc.update(pred, target)
    
    # Compute metrics
    metrics = metrics_calc.compute()
    
    # Print summary
    class_names = [
        'Trees', 'Lush Bushes', 'Grass', 'Dirt', 'Sand', 
        'Water', 'Rocks', 'Bushes', 'Mud', 'Background'
    ]
    
    print(metrics_calc.get_summary(class_names))