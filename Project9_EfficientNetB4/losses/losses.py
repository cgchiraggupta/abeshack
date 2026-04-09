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

class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss for better edge segmentation
    """
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def compute_boundary_mask(self, mask, kernel_size=5):
        """
        Compute boundary mask using morphological operations
        """
        # Convert to numpy for OpenCV operations
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask
        
        # Create binary boundary mask
        boundary_mask = np.zeros_like(mask_np)
        
        for class_idx in np.unique(mask_np):
            if class_idx == 0:  # Skip background
                continue
            
            # Create binary mask for current class
            class_mask = (mask_np == class_idx).astype(np.uint8)
            
            # Apply morphological operations to find boundaries
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            eroded = cv2.erode(class_mask, kernel, iterations=1)
            dilated = cv2.dilate(class_mask, kernel, iterations=1)
            
            # Boundary is where dilated != eroded
            class_boundary = dilated - eroded
            boundary_mask[class_boundary > 0] = 1
        
        return torch.from_numpy(boundary_mask).float()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probabilities (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            boundary_loss: Boundary-aware loss value
        """
        batch_size = pred.shape[0]
        
        # Softmax predictions
        if pred.shape[1] > 1:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = torch.sigmoid(pred)
        
        total_loss = 0
        
        for i in range(batch_size):
            # Compute boundary mask for target
            boundary_mask = self.compute_boundary_mask(target[i])
            boundary_mask = boundary_mask.to(pred.device)
            
            # Get predicted probabilities at boundary pixels
            pred_boundary = pred_soft[i] * boundary_mask.unsqueeze(0)
            
            # Calculate cross entropy at boundary
            ce_boundary = F.cross_entropy(
                pred[i].unsqueeze(0),
                target[i].unsqueeze(0),
                reduction='none'
            )
            
            # Weight by boundary mask
            weighted_ce = ce_boundary * boundary_mask.unsqueeze(0)
            
            # Normalize by number of boundary pixels
            if boundary_mask.sum() > 0:
                boundary_loss = weighted_ce.sum() / boundary_mask.sum()
            else:
                boundary_loss = weighted_ce.mean()
            
            total_loss += boundary_loss
        
        return total_loss / batch_size

class LovaszLoss(nn.Module):
    """
    Lovasz-Softmax loss for segmentation
    """
    def __init__(self):
        super(LovaszLoss, self).__init__()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            lovasz_loss: Lovasz loss value
        """
        # Flatten predictions and targets
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, pred.shape[1])
        target_flat = target.view(-1)
        
        # Calculate Lovasz loss
        loss = lovasz_softmax(pred_flat, target_flat)
        
        return loss

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
    From: https://github.com/bermanmaxim/LovaszSoftmax
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (fg.sum() == 0) and (classes not in ['all', 'present']):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    
    return mean(losses)

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    
    return vprobas, vlabels

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    
    return jaccard

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    
    for n, v in enumerate(l, 2):
        acc += v
    
    if n == 1:
        return acc
    
    return acc / n

class WeightedCombinedLoss(nn.Module):
    """
    Weighted combined loss with class-specific weights
    """
    def __init__(self, class_weights=None, loss_weights=None, device='cuda'):
        super(WeightedCombinedLoss, self).__init__()
        
        # Class weights for cross entropy
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, device=device)
        else:
            self.class_weights = None
        
        # Loss component weights
        if loss_weights is None:
            loss_weights = {
                'ce': 0.4,
                'dice': 0.3,
                'focal': 0.2,
                'boundary': 0.1
            }
        
        self.loss_weights = loss_weights
        
        # Initialize loss functions
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth masks (B, H, W) with class indices
        
        Returns:
            total_loss: Weighted combined loss
            loss_dict: Dictionary with individual loss values
        """
        # Calculate individual losses
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Combine with weights
        total_loss = (
            self.loss_weights['ce'] * ce +
            self.loss_weights['dice'] * dice +
            self.loss_weights['focal'] * focal +
            self.loss_weights['boundary'] * boundary
        )
        
        # Store loss values
        loss_dict = {
            'ce': ce.item(),
            'dice': dice.item(),
            'focal': focal.item(),
            'boundary': boundary.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict

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
    
    # Boundary Loss
    boundary_loss_fn = BoundaryLoss()
    boundary_loss = boundary_loss_fn(pred, target)
    print(f"Boundary Loss: {boundary_loss.item():.4f}")
    
    # Weighted Combined Loss
    class_weights = torch.ones(num_classes)
    weighted_loss_fn = WeightedCombinedLoss(class_weights=class_weights)
    weighted_loss, weighted_dict = weighted_loss_fn(pred, target)
    print(f"Weighted Combined Loss: {weighted_loss.item():.4f}")
    print(f"Weighted loss components: {weighted_dict}")