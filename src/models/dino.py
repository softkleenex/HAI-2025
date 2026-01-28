import torch
import torch.nn as nn
from transformers import Dinov2Model

class DeepFakeDINOv2(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", num_classes=2, freeze_backbone=False):
        super(DeepFakeDINOv2, self).__init__()
        print(f"Loading DINOv2 Model: {model_name}")
        
        self.backbone = Dinov2Model.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen.")
        else:
            print("Backbone trainable (Unfrozen).")
            
        # DINOv2 hidden sizes: small=384, base=768, large=1024, giant=1536
        hidden_size = self.backbone.config.hidden_size
        print(f"Detected hidden size: {hidden_size}")
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        cls_token = outputs.pooler_output
        logits = self.classifier(cls_token)
        return logits