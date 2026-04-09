import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPPModule, self).__init__()
        self.branches = nn.ModuleList()
        
        for rate in rates:
            if rate == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            self.branches.append(branch)
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv = nn.Conv2d(out_channels * (len(rates) + 1), out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=True)
        branch_outputs.append(global_feat)
        
        x = torch.cat(branch_outputs, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x

class MobileNetV3Segmentation(nn.Module):
    def __init__(self, num_classes=11, variant='large', pretrained=True):
        super(MobileNetV3Segmentation, self).__init__()
        
        if variant == 'large':
            mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
            backbone_features = [16, 24, 40, 112, 160]
        elif variant == 'small':
            mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
            backbone_features = [16, 16, 24, 48, 96]
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        
        self.stem = mobilenet.features[0]
        self.layer1 = mobilenet.features[1:3]
        self.layer2 = mobilenet.features[3:5]
        self.layer3 = mobilenet.features[5:7]
        self.layer4 = mobilenet.features[7:9]
        self.layer5 = mobilenet.features[9:12]
        
        self.aspp = ASPPModule(backbone_features[4], 256, rates=[1, 6, 12, 18])
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(backbone_features[3], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        x1 = self.stem(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        
        x6 = self.aspp(x6)
        
        x4 = self.decoder1(x4)
        x6 = F.interpolate(x6, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x6, x4], dim=1)
        x = self.decoder2(x)
        
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        
        return x

class MobileNetV3LightSegmentation(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super(MobileNetV3LightSegmentation, self).__init__()
        
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        backbone_features = [16, 16, 24, 48, 96]
        
        self.stem = mobilenet.features[0]
        self.layer1 = mobilenet.features[1:3]
        self.layer2 = mobilenet.features[3:5]
        self.layer3 = mobilenet.features[5:7]
        self.layer4 = mobilenet.features[7:9]
        
        self.aspp = ASPPModule(backbone_features[4], 128, rates=[1, 6, 12])
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(backbone_features[3], 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128 + 32, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        x1 = self.stem(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        
        x5 = self.aspp(x5)
        
        x4 = self.decoder1(x4)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x5, x4], dim=1)
        x = self.decoder2(x)
        
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        
        return x

class MobileNetV3WithAttention(nn.Module):
    def __init__(self, num_classes=11, variant='large', pretrained=True):
        super(MobileNetV3WithAttention, self).__init__()
        
        if variant == 'large':
            mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
            backbone_features = [16, 24, 40, 112, 160]
        else:
            mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
            backbone_features = [16, 16, 24, 48, 96]
        
        self.stem = mobilenet.features[0]
        self.layer1 = mobilenet.features[1:3]
        self.layer2 = mobilenet.features[3:5]
        self.layer3 = mobilenet.features[5:7]
        self.layer4 = mobilenet.features[7:9]
        self.layer5 = mobilenet.features[9:12]
        
        self.aspp = ASPPModule(backbone_features[4], 256, rates=[1, 6, 12, 18])
        
        self.attention1 = SqueezeExcitation(backbone_features[3])
        self.attention2 = SqueezeExcitation(256)
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(backbone_features[3], 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        x1 = self.stem(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        
        x6 = self.aspp(x6)
        x6 = self.attention2(x6)
        
        x4 = self.attention1(x4)
        x4 = self.decoder1(x4)
        
        x6 = F.interpolate(x6, size=x4.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x6, x4], dim=1)
        x = self.decoder2(x)
        
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        
        return x

if __name__ == '__main__':
    model_large = MobileNetV3Segmentation(num_classes=11, variant='large', pretrained=True)
    print(f"MobileNetV3 Large parameters: {sum(p.numel() for p in model_large.parameters()) / 1e6:.2f}M")
    
    model_small = MobileNetV3Segmentation(num_classes=11, variant='small', pretrained=True)
    print(f"MobileNetV3 Small parameters: {sum(p.numel() for p in model_small.parameters()) / 1e6:.2f}M")
    
    model_light = MobileNetV3LightSegmentation(num_classes=11, pretrained=True)
    print(f"MobileNetV3 Light parameters: {sum(p.numel() for p in model_light.parameters()) / 1e6:.2f}M")
    
    model_attention = MobileNetV3WithAttention(num_classes=11, variant='large', pretrained=True)
    print(f"MobileNetV3 with Attention parameters: {sum(p.numel() for p in model_attention.parameters()) / 1e6:.2f}M")
    
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model_large(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")