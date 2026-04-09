import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50

class FCNWrapper(nn.Module):
    def __init__(self, num_classes):
        super(FCNWrapper, self).__init__()
        self.model = fcn_resnet50(pretrained=True, progress=True)
        
        # Modify the classifier for our number of classes
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self, x):
        return self.model(x)['out']

def get_model(num_classes):
    """
    Returns a FCN model with a ResNet50 backbone from torchvision.
    """
    model = FCNWrapper(num_classes)
    return model