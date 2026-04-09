import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([
            self._make_stage(features, size) for size in sizes
        ])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()
    
    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(features)
        return nn.Sequential(prior, conv, bn, self.relu)
    
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True) 
                 for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPNet(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet101', pretrained=True):
        super(PSPNet, self).__init__()
        
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            resnet = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        if backbone == 'resnet50':
            feature_dim = 2048
        elif backbone == 'resnet101':
            feature_dim = 2048
        elif backbone == 'resnet152':
            feature_dim = 2048
        
        self.psp = PSPModule(feature_dim, 512, sizes=(1, 2, 3, 6))
        
        self.drop_1 = nn.Dropout2d(p=0.3)
        
        self.up_1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
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
        x_size = x.size()
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)
        
        x = self.psp(x)
        x = self.drop_1(x)
        x = self.up_1(x)
        
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        
        if self.training:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=x_size[2:], mode='bilinear', align_corners=True)
            return x, aux
        else:
            return x

class PSPNetLight(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet18', pretrained=True):
        super(PSPNetLight, self).__init__()
        
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        else:
            raise ValueError(f"Unsupported light backbone: {backbone}")
        
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        self.psp = PSPModule(feature_dim, 256, sizes=(1, 2, 3, 6))
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
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
        x_size = x.size()
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.psp(x)
        x = self.final_conv(x)
        
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        
        return x

class PSPNetWithAttention(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet101', pretrained=True):
        super(PSPNetWithAttention, self).__init__()
        
        if backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone for attention: {backpoint}")
        
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation = (4, 4)
                m.padding = (4, 4)
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        
        self.psp = PSPModule(feature_dim, 512, sizes=(1, 2, 3, 6))
        
        self.attention = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
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
        x_size = x.size()
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.psp(x)
        
        attention_map = self.attention(x)
        x = x * attention_map
        
        x = self.final_conv(x)
        
        x = F.interpolate(x, size=x_size[2:], mode='bilinear', align_corners=True)
        
        return x

if __name__ == '__main__':
    model = PSPNet(num_classes=11, backbone='resnet101', pretrained=True)
    print(f"PSPNet parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    dummy_input = torch.randn(2, 3, 512, 512)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output[0].shape if isinstance(output, tuple) else output.shape}")
    
    model_light = PSPNetLight(num_classes=11, backbone='resnet18', pretrained=True)
    print(f"\nPSPNetLight parameters: {sum(p.numel() for p in model_light.parameters()) / 1e6:.2f}M")
    
    model_attention = PSPNetWithAttention(num_classes=11, backbone='resnet101', pretrained=True)
    print(f"PSPNetWithAttention parameters: {sum(p.numel() for p in model_attention.parameters()) / 1e6:.2f}M")