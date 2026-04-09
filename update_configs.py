import os
import yaml

projects = {
    "Project1_DeepLabV3_ResNet101": {
        "model_type": "deeplabv3plus",
        "backbone": "resnet101"
    },
    "Project2_UNet": {
        "model_type": "unet",
        "backbone": "resnet50"
    },
    "Project3_SegFormer": {
        "model_type": "segformer",
        "backbone": "b2"
    },
    "Project4_FCN": {
        "model_type": "fcn",
        "backbone": "resnet50"
    },
    "Project5_PSPNet": {
        "model_type": "pspnet",
        "backbone": "resnet101"
    },
    "Project6_UNetPlusPlus": {
        "model_type": "unetplusplus",
        "backbone": "resnet50"
    },
    "Project7_AttentionUNet": {
        "model_type": "attention_unet",
        "backbone": "custom"
    },
    "Project8_MobileNetV3": {
        "model_type": "mobilenetv3",
        "backbone": "large"
    },
    "Project9_EfficientNetB4": {
        "model_type": "efficientnet",
        "backbone": "b4"
    },
    "Project10_DeepLabV3_ResNet50": {
        "model_type": "deeplabv3plus",
        "backbone": "resnet50"
    }
}

for project_name, model_info in projects.items():
    config_path = os.path.join(project_name, "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add model-specific configuration
        config['model']['type'] = model_info['model_type']
        config['model']['backbone'] = model_info['backbone']
        
        # Update training config for Project 1 compatibility
        if project_name == "Project1_DeepLabV3_ResNet101":
            config['training']['batch_size'] = 4
            config['training']['epochs'] = 40
            config['training']['learning_rate'] = 0.0001
            config['training']['early_stopping_patience'] = 8
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"Updated {config_path}")
    else:
        print(f"Config not found: {config_path}")

print("\nAll config files updated successfully!")