import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor

class DeepFakeViT(nn.Module):
    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model", num_classes=2):
        super(DeepFakeViT, self).__init__()
        print(f"Loading Hugging Face ViT model: {model_name}")
        self.model = ViTForImageClassification.from_pretrained(model_name)
        
        # Check if num_classes matches the pretrained model
        if self.model.classifier.out_features != num_classes:
            print(f"Warning: Model has {self.model.classifier.out_features} classes, but config says {num_classes}. Replacing head.")
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
            
    def forward(self, x):
        # HuggingFace models return a sequence classification object
        outputs = self.model(x)
        return outputs.logits

if __name__ == '__main__':
    model = DeepFakeViT()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
