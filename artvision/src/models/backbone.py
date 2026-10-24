import torch
import torch.nn as nn
import timm

class FeatureExtractor(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # num_classes=0 means it outputs the pooled embedding directly
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # We also want to expose the last conv block for Grad-CAM later
        # In timm efficientnets, this is usually accessed via features/forward_features
        # But `forward_features` returns the spatial tensor before pooling.
        
    def forward(self, x):
        # Return pooled embeddings
        return self.backbone(x)
        
    def forward_features(self, x):
        # Return unpooled feature map for Grad-CAM
        return self.backbone.forward_features(x)
        
    def get_embedding_dim(self):
        return self.backbone.num_features
