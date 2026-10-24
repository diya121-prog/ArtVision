import torch
import torch.nn as nn
from .backbone import FeatureExtractor
from .heads import StyleClassifierHead

class StyleModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = FeatureExtractor(backbone_name, pretrained=pretrained)
        embed_dim = self.backbone.get_embedding_dim()
        self.head = StyleClassifierHead(embed_dim, num_classes)
        
    def forward(self, x, return_embeddings=False):
        embeddings = self.backbone(x)
        logits = self.head(embeddings)
        if return_embeddings:
            return logits, embeddings
        return logits
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
