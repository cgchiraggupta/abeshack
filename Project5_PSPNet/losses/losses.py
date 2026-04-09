import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        dice_loss = 0
        num_classes = pred.shape[1]
        
        for class_idx in range(num_classes):
            pred_channel = pred[:, class_idx, :, :]
            target_channel = (target == class_idx).float()
            
            intersection = (pred_channel * target_channel).sum()
            union = pred_channel.sum() + target_channel.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            dice_loss += 1 - dice
        
        return dice_loss / num_classes

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        tversky_loss = 0
        num_classes = pred.shape[1]
        
        for class_idx in range(num_classes):
            pred_channel = pred[:, class_idx, :, :]
            target_channel = (target == class_idx).float()
            
            true_positives = (pred_channel * target_channel).sum()
            false_positives = (pred_channel * (1 - target_channel)).sum()
            false_negatives = ((1 - pred_channel) * target_channel).sum()
            
            tversky = (true_positives + self.smooth) / \
                     (true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth)
            
            tversky_loss += 1 - tversky
        
        return tversky_loss / num_classes

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def compute_sdf(self, mask):
        device = mask.device
        mask = mask.float()
        
        boundary = torch.zeros_like(mask)
        boundary[:, 1:, :] += torch.abs(mask[:, 1:, :] - mask[:, :-1, :])
        boundary[:, :-1, :] += torch.abs(mask[:, :-1, :] - mask[:, 1:, :])
        boundary[:, :, 1:] += torch.abs(mask[:, :, 1:] - mask[:, :, :-1])
        boundary[:, :, :-1] += torch.abs(mask[:, :, :-1] - mask[:, :, 1:])
        boundary = (boundary > 0).float()
        
        sdf = torch.zeros_like(mask)
        for b in range(mask.shape[0]):
            pos_mask = mask[b].cpu().numpy()
            pos_boundary = boundary[b].cpu().numpy()
            
            if pos_mask.sum() == 0:
                sdf[b] = self.theta0
                continue
            
            from scipy.ndimage import distance_transform_edt
            sdf_pos = distance_transform_edt(1 - pos_mask)
            sdf_neg = distance_transform_edt(pos_mask)
            
            sdf_total = sdf_pos - sdf_neg
            sdf_total = np.clip(sdf_total, -self.theta, self.theta)
            sdf_total = sdf_total / self.theta
            
            sdf[b] = torch.from_numpy(sdf_total).to(device)
        
        return sdf
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        boundary_loss = 0
        num_classes = pred.shape[1]
        
        for class_idx in range(num_classes):
            pred_channel = pred[:, class_idx, :, :]
            target_channel = (target == class_idx).float()
            
            pred_sdf = self.compute_sdf(pred_channel)
            target_sdf = self.compute_sdf(target_channel)
            
            boundary_loss += F.l1_loss(pred_sdf, target_sdf, reduction='mean')
        
        return boundary_loss / num_classes

class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()
    
    def lovasz_softmax(self, pred, target):
        pred = F.softmax(pred, dim=1)
        losses = []
        
        for c in range(pred.shape[1]):
            fg = (target == c).float()
            if fg.sum() == 0:
                continue
            
            errors = (fg - pred[:, c]).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm]
            
            grad = fg_sorted.sum()
            losses.append(torch.dot(errors_sorted, grad))
        
        return sum(losses) / pred.shape[0] if losses else torch.tensor(0.0, device=pred.device)
    
    def forward(self, pred, target):
        return self.lovasz_softmax(pred, target)

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.3, focal_weight=0.3, 
                 tversky_weight=0.2, boundary_weight=0.1, lovasz_weight=0.1):
        super(CombinedLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.lovasz_weight = lovasz_weight
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()
        self.boundary_loss = BoundaryLoss()
        self.lovasz_loss = LovaszLoss()
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        lovasz = self.lovasz_loss(pred, target)
        
        total_loss = (
            ce +
            self.dice_weight * dice +
            self.focal_weight * focal +
            self.tversky_weight * tversky +
            self.boundary_weight * boundary +
            self.lovasz_weight * lovasz
        )
        
        return total_loss

class ClassBalancedLoss(nn.Module):
    def __init__(self, class_frequencies, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        effective_num = 1.0 - np.power(beta, class_frequencies)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(class_frequencies)
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred, target):
        if self.weights.device != pred.device:
            self.weights = self.weights.to(pred.device)
        
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        weights = self.weights[target]
        weighted_loss = weights * ce_loss
        
        return weighted_loss.mean()

class OHEMLoss(nn.Module):
    def __init__(self, ratio=0.7):
        super(OHEMLoss, self).__init__()
        self.ratio = ratio
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        num_samples = int(ce_loss.numel() * self.ratio)
        hard_samples = torch.topk(ce_loss.flatten(), num_samples)[0]
        
        return hard_samples.mean()

def compute_class_weights(mask_dir, num_classes=11):
    import os
    from glob import glob
    import cv2
    from tqdm import tqdm
    
    mask_paths = glob(os.path.join(mask_dir, '*.png'))
    if not mask_paths:
        mask_paths = glob(os.path.join(mask_dir, '*.jpg'))
    
    class_counts = np.zeros(num_classes, dtype=np.float64)
    
    print("Computing class weights...")
    for mask_path in tqdm(mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        unique_values = np.unique(mask)
        for val in unique_values:
            class_idx = val // 100
            if class_idx < num_classes:
                class_counts[class_idx] += np.sum(mask == val)
    
    total_pixels = np.sum(class_counts)
    class_frequencies = class_counts / total_pixels
    
    median_freq = np.median(class_frequencies[class_frequencies > 0])
    class_weights = median_freq / class_frequencies
    class_weights[class_frequencies == 0] = 0
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print("Class frequencies:", class_frequencies)
    print("Class weights:", class_weights)
    
    return class_weights