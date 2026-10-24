import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Hook-based Grad-CAM implementation for MultiTaskModel.
    Supports targeting any task head (style, artist, genre, mood, authenticity).
    Features context-managed hook registration to prevent memory and hook leaks.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        self.model = model
        if target_layer is None:
            # Default to EfficientNet-B0 last convolutional layer
            if hasattr(model, "backbone") and hasattr(model.backbone, "backbone"):
                if hasattr(model.backbone.backbone, "conv_head"):
                    target_layer = model.backbone.backbone.conv_head
                else:
                    # Find last Conv2d in backbone
                    convs = [m for m in model.backbone.backbone.modules() if isinstance(m, nn.Conv2d)]
                    target_layer = convs[-1] if convs else None
            elif hasattr(model, "features"):
                convs = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
                target_layer = convs[-1] if convs else None

        if target_layer is None:
            raise ValueError("Could not automatically locate the target convolutional layer for Grad-CAM.")

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        self.remove_hooks()
        h1 = self.target_layer.register_forward_hook(self._forward_hook)
        h2 = self.target_layer.register_full_backward_hook(self._backward_hook)
        self.hooks = [h1, h2]

    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks = []
        self.activations = None
        self.gradients = None

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        task: str = "style",
        target_class: int = None,
        retain_graph: bool = False
    ) -> np.ndarray:
        """
        Computes Grad-CAM attribution heatmap for a single image tensor [1, C, H, W].
        Returns normalized 2D numpy array [H, W] with values in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        input_tensor.requires_grad_(True)
        outputs = self.model(input_tensor)

        if task not in outputs:
            raise KeyError(f"Task '{task}' not found in model outputs: {list(outputs.keys())}")

        task_out = outputs[task]

        # Determine target scalar to backpropagate
        if task == "authenticity":
            # Binary logit
            logit = task_out.squeeze()
            if target_class is None or target_class == 1:
                # Target AI
                target_scalar = logit
            else:
                # Target Human
                target_scalar = -logit
        else:
            # Multi-class classification
            if target_class is None:
                target_class = int(torch.argmax(task_out, dim=-1).item())
            target_scalar = task_out[0, target_class]

        # Backward pass
        target_scalar.backward(retain_graph=retain_graph)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        # Channel importance weights: Global average pooling over spatial dimensions
        # gradients: [1, C, H_feat, W_feat]
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True) # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True) # [1, 1, H_feat, W_feat]

        # Apply ReLU to focus on positive attribution
        cam = F.relu(cam)

        # Upsample to input tensor spatial resolution
        h_in, w_in = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h_in, w_in), mode="bilinear", align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalize to [0, 1]
        c_min = float(cam_np.min())
        c_max = float(cam_np.max())
        if c_max - c_min > 1e-8:
            cam_np = (cam_np - c_min) / (c_max - c_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np


def overlay_heatmap(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> Image.Image:
    """
    Overlays 2D [0, 1] heatmap onto original PIL Image.
    """
    orig_w, orig_h = original_image.size
    # Resize heatmap to original image dimensions
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))
    heatmap_uint8 = (heatmap_resized * 255.0).astype(np.uint8)

    # Apply OpenCV Colormap
    colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original_image.convert("RGB"))
    blended = (alpha * colored_cam + (1.0 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)


def calculate_border_mass_fraction(heatmap: np.ndarray, border_ratio: float = 0.1) -> float:
    """
    Calculates the proportion of total attribution mass falling in the outer border region.
    """
    h, w = heatmap.shape
    total_mass = float(np.sum(heatmap))
    if total_mass <= 1e-8:
        return 0.0

    border_h = max(1, int(round(h * border_ratio)))
    border_w = max(1, int(round(w * border_ratio)))

    # Mask interior
    mask = np.ones((h, w), dtype=bool)
    mask[border_h : h - border_h, border_w : w - border_w] = False

    border_mass = float(np.sum(heatmap[mask]))
    return float(border_mass / total_mass)
