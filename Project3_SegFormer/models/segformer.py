from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

class SegFormerWrapper(nn.Module):
    def __init__(self, num_classes):
        super(SegFormerWrapper, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

def get_model(num_classes):
    """
    Returns a SegFormer-B2 model from HuggingFace transformers.
    """
    model = SegFormerWrapper(num_classes)
    return model