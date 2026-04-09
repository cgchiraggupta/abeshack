import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DeepLabV3ResNet50(nn.Module):
    """
    DeepLabV3+ with ResNet50 backbone for off-road terrain segmentation
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(DeepLabV3ResNet50, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Atrous Spatial Pyramid Pooling (ASPP)
        self.aspp = ASPP(2048, 256)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Segmentation logits (B, num_classes, H, W)
        """
        # Get features from ResNet backbone
        x = self.layer0(x)      # (B, 64, H/4, W/4)
        low_level = self.layer1(x)  # (B, 256, H/4, W/4) - save for decoder
        
        x = self.layer2(low_level)  # (B, 512, H/8, W/8)
        x = self.layer3(x)          # (B, 1024, H/16, W/16)
        x = self.layer4(x)          # (B, 2048, H/16, W/16)
        
        # Apply ASPP
        x = self.aspp(x)            # (B, 256, H/16, W/16)
        
        # Process low-level features
        low_level = self.low_level_conv(low_level)  # (B, 48, H/4, W/4)
        
        # Upsample ASPP features to match low-level feature size
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with low-level features
        x = torch.cat([x, low_level], dim=1)  # (B, 304, H/4, W/4)
        
        # Apply decoder
        x = self.decoder(x)  # (B, num_classes, H/4, W/4)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)
        
        return x

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module
    """
    def __init__(self, in_channels, out_channels=256):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=6
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=18
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x):
        # Branch 1: 1x1 convolution
        x1 = self.conv1(x)
        
        # Branch 2: 3x3 convolution with rate=6
        x2 = self.conv2(x)
        
        # Branch 3: 3x3 convolution with rate=12
        x3 = self.conv3(x)
        
        # Branch 4: 3x3 convolution with rate=18
        x4 = self.conv4(x)
        
        # Branch 5: Global average pooling
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class DeepLabV3ResNet50Simple(nn.Module):
    """
    Simplified DeepLabV3+ with ResNet50 (no ASPP, faster inference)
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(DeepLabV3ResNet50Simple, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the last two layers (avgpool and fc)
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Get features from ResNet backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*16, x.shape[3]*16), mode='bilinear', align_corners=False)
        
        return x

class DeepLabV3ResNet50Lite(nn.Module):
    """
    Lite version of DeepLabV3+ with fewer parameters
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(DeepLabV3ResNet50Lite, self).__init__()
        
        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Use only first 3 blocks
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        
        # Lite ASPP
        self.aspp = LiteASPP(1024, 128)
        
        # Lite decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Get features from ResNet backbone
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Apply ASPP
        x = self.aspp(x)
        
        # Apply decoder
        x = self.decoder(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*8, x.shape[3]*8), mode='bilinear', align_corners=False)
        
        return x

class LiteASPP(nn.Module):
    """
    Lite version of ASPP with fewer branches
    """
    def __init__(self, in_channels, out_channels=128):
        super(LiteASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=6
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 convolution with rate=12
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Branch 1: 1x1 convolution
        x1 = self.conv1(x)
        
        # Branch 2: 3x3 convolution with rate=6
        x2 = self.conv2(x)
        
        # Branch 3: 3x3 convolution with rate=12
        x3 = self.conv3(x)
        
        # Concatenate all branches
        x = torch.cat([x1, x2, x3], dim=1)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

# Utility function to create model
def create_deeplabv3_model(model_type='standard', num_classes=10, pretrained=True):
    """
    Factory function to create different DeepLabV3+ variants
    
    Args:
        model_type: Type of model to create
            'standard': Standard DeepLabV3+ with ResNet50
            'simple': Simplified version without ASPP
            'lite': Lite version with fewer parameters
        num_classes: Number of output classes
        pretrained: Use pretrained weights
    
    Returns:
        model: Created model
    """
    if model_type == 'standard':
        return DeepLabV3ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'simple':
        return DeepLabV3ResNet50Simple(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'lite':
        return DeepLabV3ResNet50Lite(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage and testing
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create standard model
    model = DeepLabV3ResNet50(num_classes=10, pretrained=True)
    model.to(device)
    
    # Create sample input
    batch_size = 2
    x = torch.randn(batch_size, 3, 512, 512).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test different model variants
    print("\nTesting different model variants:")
    
    variants = ['standard', 'simple', 'lite']
    
    for variant in variants:
        try:
            model_variant = create_deeplabv3_model(variant, num_classes=10, pretrained=False)
            model_variant.to(device)
            
            with torch.no_grad():
                output_variant = model_variant(x)
            
            print(f"{variant}: Output shape = {output_variant.shape}")
            
            # Clean up
            del model_variant
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{variant}: Error - {e}")