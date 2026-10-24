import os
import torch
import torch.nn as nn
from .backbone import FeatureExtractor


class TaskHead(nn.Module):
    """Configurable task-specific classification head."""
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_features)
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, out_features)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared EfficientNet backbone and modular task heads.
    Supports style, artist, genre, mood, authenticity, and extensible placeholders.
    """
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config.get("model", {})
        tasks_cfg = config.get("tasks", {})
        
        backbone_name = model_cfg.get("backbone", "efficientnet_b0")
        pretrained = model_cfg.get("pretrained", True)
        self.backbone = FeatureExtractor(backbone_name, pretrained=pretrained)
        self.embed_dim = self.backbone.get_embedding_dim()
        
        hidden_dim = model_cfg.get("hidden_dim", 256)
        dropout = model_cfg.get("dropout", 0.2)
        
        self.heads = nn.ModuleDict()
        self.enabled_tasks = []
        
        for task_name, task_info in tasks_cfg.items():
            if task_info.get("enabled", False):
                num_classes = task_info.get("num_classes", 1)
                if task_name == "artist":
                    # Artist head uses top_n classes
                    num_classes = task_info.get("top_n", 50)
                elif task_name == "authenticity":
                    # Binary classification: 1 logit for BCEWithLogitsLoss
                    num_classes = 1
                
                self.heads[task_name] = TaskHead(
                    in_features=self.embed_dim,
                    out_features=num_classes,
                    hidden_dim=hidden_dim,
                    dropout=dropout
                )
                self.enabled_tasks.append(task_name)
                
        # Optional backbone initialization from Phase 3 checkpoint
        ckpt_path = model_cfg.get("pretrained_checkpoint", None)
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_backbone_from_checkpoint(ckpt_path)

    def forward(self, x: torch.Tensor, return_embeddings: bool = False):
        embeddings = self.backbone(x)
        outputs = {}
        for task_name, head in self.heads.items():
            out = head(embeddings)
            if task_name == "authenticity":
                out = out.squeeze(-1)  # (B, 1) -> (B,)
            outputs[task_name] = out
            
        if return_embeddings:
            return outputs, embeddings
        return outputs

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Expose unpooled spatial feature maps for Grad-CAM."""
        return self.backbone.forward_features(x)

    def get_embedding_dim(self) -> int:
        return self.embed_dim

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def load_backbone_from_checkpoint(self, checkpoint_path: str):
        """Loads backbone weights from a single-task style checkpoint."""
        print(f"Loading backbone weights from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                # e.g., 'backbone.backbone.conv_stem.weight' -> 'backbone.conv_stem.weight'
                clean_key = k[len("backbone."):]
                backbone_dict[clean_key] = v
                
        if backbone_dict:
            msg = self.backbone.load_state_dict(backbone_dict, strict=False)
            print(f"Backbone initialized with status: {msg}")
        else:
            print("No matching backbone keys found in checkpoint.")
