import torch

def dice_score(preds, targets, num_classes=10, smooth=1e-6):
    """
    Computes the Dice score for multiclass segmentation.
    Expects preds: (B, C, H, W) and targets: (B, H, W)
    """
    preds = torch.softmax(preds, dim=1)
    
    # One-hot encode targets to (B, C, H, W)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (preds * targets_one_hot).sum(dim=(0, 2, 3))
    union = preds.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean(), dice

def iou_score(preds, targets, num_classes=10, smooth=1e-6):
    """
    Computes the Intersection over Union (IoU) for multiclass segmentation.
    Expects preds: (B, C, H, W) and targets: (B, H, W)
    """
    preds = torch.softmax(preds, dim=1)
    
    # One-hot encode targets to (B, C, H, W)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    
    intersection = (preds * targets_one_hot).sum(dim=(0, 2, 3))
    total = preds.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
    union = total - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()
