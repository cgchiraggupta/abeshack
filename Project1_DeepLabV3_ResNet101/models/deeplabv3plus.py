import segmentation_models_pytorch as smp

def get_model(num_classes):
    """
    Returns a DeepLabV3+ model with a ResNet101 backbone.
    """
    model = smp.DeepLabV3Plus(
        encoder_name="resnet101",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    return model
