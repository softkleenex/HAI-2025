import torch
import torch.nn as nn
import timm

class DeepFakeClassifier(nn.Module):
    def __init__(self, backbone='efficientnet_b0', pretrained=True, num_classes=2, dropout=0.2):
        super(DeepFakeClassifier, self).__init__()
        print(f"Creating model: {backbone} (pretrained={pretrained})")
        
        # timm.create_model automatically handles the classifier head replacement
        # if num_classes is specified.
        self.model = timm.create_model(
            backbone, 
            pretrained=pretrained, 
            num_classes=num_classes,
            drop_rate=dropout
        )
        
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Test with ConvNeXt
    try:
        model = DeepFakeClassifier(backbone='convnext_tiny', pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        print(f"ConvNeXt Output shape: {y.shape}")
    except Exception as e:
        print(f"Error: {e}")