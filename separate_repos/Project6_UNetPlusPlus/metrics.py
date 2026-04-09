import torch
import torch.nn as nn
import numpy as np

def dice_score(pred, target, num_classes, smooth=1e-6):
    dice_scores = torch.zeros(num_classes, device=pred.device)
    
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores[class_idx] = dice
    
    return dice_scores

def iou_score(pred, target, num_classes, smooth=1e-6):
    iou_scores = torch.zeros(num_classes, device=pred.device)
    
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        
        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores[class_idx] = iou
    
    return iou_scores

def precision_score(pred, target, num_classes, smooth=1e-6):
    precision_scores = torch.zeros(num_classes, device=pred.device)
    
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        
        true_positives = (pred_mask * target_mask).sum()
        predicted_positives = pred_mask.sum()
        
        precision = (true_positives + smooth) / (predicted_positives + smooth)
        precision_scores[class_idx] = precision
    
    return precision_scores

def recall_score(pred, target, num_classes, smooth=1e-6):
    recall_scores = torch.zeros(num_classes, device=pred.device)
    
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        
        true_positives = (pred_mask * target_mask).sum()
        actual_positives = target_mask.sum()
        
        recall = (true_positives + smooth) / (actual_positives + smooth)
        recall_scores[class_idx] = recall
    
    return recall_scores

def f1_score(pred, target, num_classes, smooth=1e-6):
    precision = precision_score(pred, target, num_classes, smooth)
    recall = recall_score(pred, target, num_classes, smooth)
    
    f1_scores = (2 * precision * recall) / (precision + recall + smooth)
    return f1_scores

def accuracy_score(pred, target):
    correct = (pred == target).float().sum()
    total = target.numel()
    return correct / total

def compute_confusion_matrix(pred, target, num_classes):
    confusion_matrix = torch.zeros((num_classes, num_classes), device=pred.device, dtype=torch.long)
    
    for i in range(num_classes):
        for j in range(num_classes):
            confusion_matrix[i, j] = ((pred == i) & (target == j)).sum()
    
    return confusion_matrix

def compute_mean_iou(confusion_matrix, smooth=1e-6):
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=1) + confusion_matrix.sum(dim=0) - intersection
    iou = (intersection + smooth) / (union + smooth)
    mean_iou = iou[~torch.isnan(iou)].mean()
    return mean_iou

def compute_pixel_accuracy(confusion_matrix):
    correct = torch.diag(confusion_matrix).sum()
    total = confusion_matrix.sum()
    return correct / total

def compute_class_accuracy(confusion_matrix, smooth=1e-6):
    class_acc = torch.diag(confusion_matrix) / (confusion_matrix.sum(dim=1) + smooth)
    return class_acc

def compute_kappa_score(confusion_matrix):
    total = confusion_matrix.sum()
    observed_accuracy = torch.diag(confusion_matrix).sum() / total
    
    expected_accuracy = 0
    for i in range(confusion_matrix.shape[0]):
        row_sum = confusion_matrix[i, :].sum()
        col_sum = confusion_matrix[:, i].sum()
        expected_accuracy += (row_sum * col_sum)
    expected_accuracy /= (total * total)
    
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy + 1e-6)
    return kappa

def compute_dice_coefficient_batch(preds, targets, num_classes, smooth=1e-6):
    batch_size = preds.shape[0]
    dice_scores = torch.zeros((batch_size, num_classes), device=preds.device)
    
    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
            pred_mask = (preds[batch_idx] == class_idx).float()
            target_mask = (targets[batch_idx] == class_idx).float()
            
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum()
            
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores[batch_idx, class_idx] = dice
    
    return dice_scores

def compute_iou_batch(preds, targets, num_classes, smooth=1e-6):
    batch_size = preds.shape[0]
    iou_scores = torch.zeros((batch_size, num_classes), device=preds.device)
    
    for batch_idx in range(batch_size):
        for class_idx in range(num_classes):
            pred_mask = (preds[batch_idx] == class_idx).float()
            target_mask = (targets[batch_idx] == class_idx).float()
            
            intersection = (pred_mask * target_mask).sum()
            union = pred_mask.sum() + target_mask.sum() - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            iou_scores[batch_idx, class_idx] = iou
    
    return iou_scores

def compute_boundary_iou(pred, target, num_classes, dilation_radius=1):
    import torch.nn.functional as F
    
    boundary_ious = torch.zeros(num_classes, device=pred.device)
    
    for class_idx in range(num_classes):
        pred_mask = (pred == class_idx).float()
        target_mask = (target == class_idx).float()
        
        if pred_mask.sum() == 0 and target_mask.sum() == 0:
            boundary_ious[class_idx] = 1.0
            continue
        
        kernel = torch.ones(1, 1, 2*dilation_radius+1, 2*dilation_radius+1, device=pred.device)
        
        pred_dilated = F.conv2d(pred_mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation_radius)
        pred_eroded = F.conv2d(pred_mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation_radius)
        pred_boundary = (pred_dilated > 0).float() - (pred_eroded == kernel.numel()).float()
        
        target_dilated = F.conv2d(target_mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation_radius)
        target_eroded = F.conv2d(target_mask.unsqueeze(0).unsqueeze(0), kernel, padding=dilation_radius)
        target_boundary = (target_dilated > 0).float() - (target_eroded == kernel.numel()).float()
        
        intersection = (pred_boundary * target_boundary).sum()
        union = pred_boundary.sum() + target_boundary.sum() - intersection
        
        if union > 0:
            boundary_ious[class_idx] = intersection / union
        else:
            boundary_ious[class_idx] = 0.0
    
    return boundary_ious

def compute_hausdorff_distance(pred, target, num_classes):
    from scipy.spatial.distance import directed_hausdorff
    
    hausdorff_distances = torch.zeros(num_classes, device=pred.device)
    
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    for class_idx in range(num_classes):
        pred_coords = np.argwhere(pred_np == class_idx)
        target_coords = np.argwhere(target_np == class_idx)
        
        if len(pred_coords) == 0 or len(target_coords) == 0:
            hausdorff_distances[class_idx] = float('inf')
            continue
        
        hausdorff = max(
            directed_hausdorff(pred_coords, target_coords)[0],
            directed_hausdorff(target_coords, pred_coords)[0]
        )
        hausdorff_distances[class_idx] = hausdorff
    
    return hausdorff_distances

class SegmentationMetrics:
    def __init__(self, num_classes, device='cuda'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
    
    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_classes, self.num_classes), 
                                           device=self.device, dtype=torch.long)
        self.total_samples = 0
    
    def update(self, pred, target):
        batch_size = pred.shape[0]
        self.total_samples += batch_size
        
        for i in range(batch_size):
            pred_flat = pred[i].flatten()
            target_flat = target[i].flatten()
            
            for true_class in range(self.num_classes):
                for pred_class in range(self.num_classes):
                    self.confusion_matrix[true_class, pred_class] += \
                        ((target_flat == true_class) & (pred_flat == pred_class)).sum().item()
    
    def compute(self):
        metrics = {}
        
        intersection = torch.diag(self.confusion_matrix).float()
        union = self.confusion_matrix.sum(dim=1).float() + self.confusion_matrix.sum(dim=0).float() - intersection
        
        metrics['iou'] = intersection / (union + 1e-6)
        metrics['mean_iou'] = metrics['iou'][~torch.isnan(metrics['iou'])].mean()
        
        metrics['dice'] = (2 * intersection) / (self.confusion_matrix.sum(dim=1).float() + 
                                               self.confusion_matrix.sum(dim=0).float() + 1e-6)
        metrics['mean_dice'] = metrics['dice'][~torch.isnan(metrics['dice'])].mean()
        
        metrics['pixel_accuracy'] = intersection.sum() / self.confusion_matrix.sum()
        
        metrics['class_accuracy'] = intersection / (self.confusion_matrix.sum(dim=1).float() + 1e-6)
        metrics['mean_class_accuracy'] = metrics['class_accuracy'][~torch.isnan(metrics['class_accuracy'])].mean()
        
        return metrics
    
    def get_confusion_matrix(self):
        return self.confusion_matrix.cpu().numpy()