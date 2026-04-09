import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm

class EfficientNetB4Segmentation(nn.Module):
    """
    EfficientNet-B4 based segmentation model for off-road terrain
    Uses pretrained EfficientNet-B4 as encoder with custom decoder
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB4Segmentation, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.encoder = timm.create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        
        # Get channel dimensions from EfficientNet-B4
        # EfficientNet-B4 feature channels: [24, 32, 56, 160, 448]
        self.encoder_channels = [24, 32, 56, 160, 448]
        
        # Decoder channels
        self.decoder_channels = [256, 128, 64, 32, 16]
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # Build decoder
        for i in range(len(self.encoder_channels) - 1, 0, -1):
            in_channels = self.encoder_channels[i] + self.decoder_channels[i-1] if i < len(self.encoder_channels) - 1 else self.encoder_channels[i]
            out_channels = self.decoder_channels[i-1]
            
            self.decoder_blocks.append(
                DecoderBlock(in_channels, out_channels)
            )
        
        # Final convolution to get segmentation mask
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.decoder_channels[0], 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, kernel_size=1)
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
        # Get encoder features
        encoder_features = self.encoder(x)
        
        # Reverse encoder features for decoder
        encoder_features = encoder_features[::-1]
        
        # Start with the deepest feature map
        x = encoder_features[0]
        
        # Decoder with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            # Skip connection from encoder
            skip = encoder_features[i + 1] if i + 1 < len(encoder_features) else None
            
            # Apply decoder block
            x = decoder_block(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)
        
        return x

class DecoderBlock(nn.Module):
    """
    Decoder block with skip connection
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x, skip=None):
        """
        Forward pass with optional skip connection
        
        Args:
            x: Input tensor
            skip: Skip connection tensor (optional)
        
        Returns:
            Output tensor
        """
        # Upsample
        x = self.up(x)
        
        # Concatenate with skip connection if provided
        if skip is not None:
            # Adjust skip connection size if needed
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)
        
        # First convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        # Dropout
        x = self.dropout(x)
        
        return x

class EfficientNetB4DeepLab(nn.Module):
    """
    EfficientNet-B4 with DeepLabV3+ style decoder
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB4DeepLab, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.encoder = timm.create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        
        # Get feature channels
        self.encoder_channels = [24, 32, 56, 160, 448]
        
        # ASPP module
        self.aspp = ASPP(self.encoder_channels[-1], 256)
        
        # Low-level feature processing
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(self.encoder_channels[1], 48, kernel_size=1),
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
    
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        
        # Low-level features (from early layer)
        low_level = self.low_level_conv(features[1])
        
        # High-level features with ASPP
        high_level = self.aspp(features[-1])
        
        # Upsample high-level features
        high_level = F.interpolate(high_level, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with low-level features
        x = torch.cat([high_level, low_level], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
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

class EfficientNetB4UNet(nn.Module):
    """
    EfficientNet-B4 with UNet-like decoder
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB4UNet, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.encoder = timm.create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        
        # Encoder channels
        self.encoder_channels = [24, 32, 56, 160, 448]
        
        # Decoder
        self.up1 = UpBlock(448, 160)
        self.up2 = UpBlock(320, 56)  # 160*2 = 320
        self.up3 = UpBlock(112, 32)  # 56*2 = 112
        self.up4 = UpBlock(64, 24)   # 32*2 = 64
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        
        # Decoder with skip connections
        x = self.up1(features[4], features[3])
        x = self.up2(x, features[2])
        x = self.up3(x, features[1])
        x = self.up4(x, features[0])
        
        # Final convolution
        x = self.final_conv(x)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)
        
        return x

class UpBlock(nn.Module):
    """
    Upsampling block for UNet-like architecture
    """
    def __init__(self, in_channels, skip_channels):
        super(UpBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Convolution for skip connection
        self.skip_conv = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True)
        )
        
        # Main convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.Dropout2d(0.1)
        )
    
    def forward(self, x, skip):
        # Upsample
        x = self.up(x)
        
        # Process skip connection
        skip = self.skip_conv(skip)
        
        # Adjust sizes if needed
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolution block
        x = self.conv(x)
        
        return x

class EfficientNetB4FPN(nn.Module):
    """
    EfficientNet-B4 with Feature Pyramid Network (FPN) decoder
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(EfficientNetB4FPN, self).__init__()
        
        # Load pretrained EfficientNet-B4
        self.encoder = timm.create_model('efficientnet_b4', pretrained=pretrained, features_only=True)
        
        # Encoder channels
        self.encoder_channels = [24, 32, 56, 160, 448]
        
        # FPN lateral connections
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        # Build FPN
        for i in range(len(self.encoder_channels)):
            lateral_conv = nn.Conv2d(self.encoder_channels[i], 256, kernel_size=1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        
        # Feature fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256 * len(self.encoder_channels), 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        
        # Apply lateral convolutions
        lateral_features = []
        for i, (feature, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lateral = lateral_conv(feature)
            lateral_features.append(lateral)
        
        # Build FPN features (top-down)
        fpn_features = []
        prev_feature = None
        
        for i in range(len(lateral_features) - 1, -1, -1):
            lateral = lateral_features[i]
            
            if prev_feature is not None:
                # Upsample previous feature
                prev_feature = F.interpolate(prev_feature, size=lateral.shape[2:], mode='bilinear', align_corners=False)
                # Add
                lateral = lateral + prev_feature
            
            # Apply FPN convolution
            fpn = self.fpn_convs[i](lateral)
            fpn_features.append(fpn)
            
            prev_feature = fpn
        
        # Reverse to get correct order
        fpn_features = fpn_features[::-1]
        
        # Upsample all features to same size (largest)
        target_size = fpn_features[0].shape[2:]
        upsampled_features = []
        
        for fpn in fpn_features:
            if fpn.shape[2:] != target_size:
                fpn = F.interpolate(fpn, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(fpn)
        
        # Concatenate all FPN features
        fused = torch.cat(upsampled_features, dim=1)
        fused = self.fusion_conv(fused)
        
        # Apply segmentation head
        x = self.seg_head(fused)
        
        # Upsample to input size
        x = F.interpolate(x, size=(x.shape[2]*4, x.shape[3]*4), mode='bilinear', align_corners=False)
        
        return x

# Utility function to create model
def create_efficientnet_b4_model(model_type='default', num_classes=10, pretrained=True):
    """
    Factory function to create different EfficientNet-B4 based models
    
    Args:
        model_type: Type of model to create
            'default': Standard EfficientNet-B4 with custom decoder
            'deeplab': EfficientNet-B4 with DeepLabV3+ decoder
            'unet': EfficientNet-B4 with UNet decoder
            'fpn': EfficientNet-B4 with FPN decoder
        num_classes: Number of output classes
        pretrained: Use pretrained weights
    
    Returns:
        model: Created model
    """
    if model_type == 'default':
        return EfficientNetB4Segmentation(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'deeplab':
        return EfficientNetB4DeepLab(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'unet':
        return EfficientNetB4UNet(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'fpn':
        return EfficientNetB4FPN(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage and testing
if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EfficientNetB4Segmentation(num_classes=10, pretrained=True)
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
    
    variants = ['default', 'deeplab', 'unet', 'fpn']
    
    for variant in variants:
        try:
            model_variant = create_efficientnet_b4_model(variant, num_classes=10, pretrained=False)
            model_variant.to(device)
            
            with torch.no_grad():
                output_variant = model_variant(x)
            
            print(f"{variant}: Output shape = {output_variant.shape}")
            
            # Clean up
            del model_variant
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"{variant}: Error - {e}")