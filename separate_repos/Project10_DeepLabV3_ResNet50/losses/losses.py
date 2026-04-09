import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            dice_loss: Dice loss value
        """
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax or sigmoid for pred
        if pred.shape[1] > 1:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred_soft.contiguous().view(pred_soft.shape[0], pred_soft.shape[1], -1)
        target_flat = target_onehot.contiguous().view(target_onehot.shape[0], target_onehot.shape[1], -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        
        # Calculate Dice coefficient per class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        dice_loss = 1 - dice.mean()
        
        return dice_loss

class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            iou_loss: IoU loss value
        """
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax or sigmoid for pred
        if pred.shape[1] > 1:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred_soft.contiguous().view(pred_soft.shape[0], pred_soft.shape[1], -1)
        target_flat = target_onehot.contiguous().view(target_onehot.shape[0], target_onehot.shape[1], -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection
        
        # Calculate IoU per class
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        # Average over classes and batch
        iou_loss = 1 - iou.mean()
        
        return iou_loss

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            focal_loss: Focal loss value
        """
        # Apply softmax to get probabilities
        if pred.shape[1] > 1:
            probs = F.softmax(pred, dim=1)
        else:
            probs = torch.sigmoid(pred)
            probs = torch.cat([1 - probs, probs], dim=1)
        
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate focal weight
        pt = (target_onehot * probs).sum(dim=1) + ((1 - target_onehot) * (1 - probs)).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Calculate cross entropy
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        
        # Apply focal weight
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss with alpha/beta parameters
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            tversky_loss: Tversky loss value
        """
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Softmax or sigmoid for pred
        if pred.shape[1] > 1:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = torch.sigmoid(pred)
        
        # Flatten tensors
        pred_flat = pred_soft.contiguous().view(pred_soft.shape[0], pred_soft.shape[1], -1)
        target_flat = target_onehot.contiguous().view(target_onehot.shape[0], target_onehot.shape[1], -1)
        
        # Calculate true positives, false positives, false negatives
        tp = (pred_flat * target_flat).sum(dim=2)
        fp = (pred_flat * (1 - target_flat)).sum(dim=2)
        fn = ((1 - pred_flat) * target_flat).sum(dim=2)
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Average over classes and batch
        tversky_loss = 1 - tversky.mean()
        
        return tversky_loss

class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components
    """
    def __init__(self, weights=None, class_weights=None, device='cuda'):
        super(CombinedLoss, self).__init__()
        
        # Default weights for each loss component
        if weights is None:
            weights = {
                'ce': 0.4,
                'dice': 0.3,
                'focal': 0.2,
                'tversky': 0.1
            }
        
        self.weights = weights
        
        # Initialize individual loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()
        
        self.device = device
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            combined_loss: Combined loss value
            loss_dict: Dictionary with individual loss values
        """
        # Calculate individual losses
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        # Combine losses with weights
        combined = (
            self.weights['ce'] * ce +
            self.weights['dice'] * dice +
            self.weights['focal'] * focal +
            self.weights['tversky'] * tversky
        )
        
        # Store individual losses
        loss_dict = {
            'ce': ce.item(),
            'dice': dice.item(),
            'focal': focal.item(),
            'tversky': tversky.item(),
            'total': combined.item()
        }
        
        return combined, loss_dict

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    batch_size = 2
    num_classes = 10
    height, width = 512, 512
    
    # Random predictions and targets
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test each loss function
    print("Testing loss functions...")
    
    # Dice Loss
    dice_loss_fn = DiceLoss()
    dice_loss = dice_loss_fn(pred, target)
    print(f"Dice Loss: {dice_loss.item():.4f}")
    
    # IoU Loss
    iou_loss_fn = IoULoss()
    iou_loss = iou_loss_fn(pred, target)
    print(f"IoU Loss: {iou_loss.item():.4f}")
    
    # Focal Loss
    focal_loss_fn = FocalLoss()
    focal_loss = focal_loss_fn(pred, target)
    print(f"Focal Loss: {focal_loss.item():.4f}")
    
    # Tversky Loss
    tversky_loss_fn = TverskyLoss()
    tversky_loss = tversky_loss_fn(pred, target)
    print(f"Tversky Loss: {tversky_loss.item():.4f}")
    
    # Combined Loss
    combined_loss_fn = CombinedLoss()
    combined_loss, loss_dict = combined_loss_fn(pred, target)
    print(f"Combined Loss: {combined_loss.item():.4f}")
    print(f"Loss components: {loss_dict}")