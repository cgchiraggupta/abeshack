import segmentation_models_pytorch as smp

def get_model(num_classes):
    """
    Returns a UNet model with a ResNet50 backbone.
    """
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    return model