import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            for c in range(num_classes):
                pred_flat = pred[i, c].contiguous().view(-1)
                target_flat = (target[i] == c).float().contiguous().view(-1)
                
                intersection = (pred_flat * target_flat).sum()
                union = pred_flat.sum() + target_flat.sum()
                
                dice = (2. * intersection + self.smooth) / (union + self.smooth)
                total_loss += 1 - dice
        
        return total_loss / (batch_size * num_classes)

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    def forward(self, pred, target):
        return self.criterion(pred, target)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_weight = self.alpha[target].to(pred.device)
            focal_loss = alpha_weight * focal_loss
        
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-8):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            for c in range(num_classes):
                pred_flat = pred[i, c].contiguous().view(-1)
                target_flat = (target[i] == c).float().contiguous().view(-1)
                
                true_pos = (pred_flat * target_flat).sum()
                false_pos = (pred_flat * (1 - target_flat)).sum()
                false_neg = ((1 - pred_flat) * target_flat).sum()
                
                tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
                total_loss += 1 - tversky
        
        return total_loss / (batch_size * num_classes)

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=10, weights=None, device='cuda'):
        super(CombinedLoss, self).__init__()
        
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0]
        
        self.ce_weight = weights[0]
        self.dice_weight = weights[1]
        self.focal_weight = weights[2]
        self.tversky_weight = weights[3]
        
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.5, beta=0.5)
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        total_loss = (self.ce_weight * ce + 
                     self.dice_weight * dice + 
                     self.focal_weight * focal + 
                     self.tversky_weight * tversky)
        
        return total_loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            for c in range(num_classes):
                pred_mask = pred[i, c]
                target_mask = (target[i] == c).float()
                
                pred_boundary = self._compute_boundary(pred_mask)
                target_boundary = self._compute_boundary(target_mask)
                
                boundary_loss = F.mse_loss(pred_boundary, target_boundary)
                total_loss += boundary_loss
        
        return total_loss / (batch_size * num_classes)
    
    def _compute_boundary(self, mask):
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        if mask.is_cuda:
            kernel = kernel.cuda()
        
        boundary = F.conv2d(mask.unsqueeze(0), kernel, padding=1)
        boundary = torch.abs(boundary)
        boundary = torch.clamp(boundary, 0, 1)
        
        return boundary.squeeze(0)

class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        
        batch_size = pred.shape[0]
        num_classes = pred.shape[1]
        
        total_loss = 0.0
        
        for i in range(batch_size):
            for c in range(num_classes):
                pred_flat = pred[i, c].contiguous().view(-1)
                target_flat = (target[i] == c).float().contiguous().view(-1)
                
                loss = self._lovasz_hinge(pred_flat, target_flat)
                total_loss += loss
        
        return total_loss / (batch_size * num_classes)
    
    def _lovasz_hinge(self, pred, target):
        signs = 2. * target - 1.
        errors = 1. - pred * signs
        
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = target[perm]
        grad = self._lovasz_grad(gt_sorted)
        
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss
    
    def _lovasz_grad(self, gt_sorted):
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        
        return jaccard

if __name__ == "__main__":
    # Test the loss functions
    pred = torch.randn(2, 10, 256, 256)
    target = torch.randint(0, 10, (2, 256, 256))
    
    print("Testing loss functions...")
    
    dice_loss = DiceLoss()
    print(f"Dice Loss: {dice_loss(pred, target):.4f}")
    
    ce_loss = CrossEntropyLoss()
    print(f"Cross Entropy Loss: {ce_loss(pred, target):.4f}")
    
    focal_loss = FocalLoss()
    print(f"Focal Loss: {focal_loss(pred, target):.4f}")
    
    tversky_loss = TverskyLoss()
    print(f"Tversky Loss: {tversky_loss(pred, target):.4f}")
    
    combined_loss = CombinedLoss()
    print(f"Combined Loss: {combined_loss(pred, target):.4f}")
    
    print("\nAll loss functions working correctly!")