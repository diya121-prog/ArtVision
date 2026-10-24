import torch
import torch.nn as nn

def get_criterion(class_counts=None, label_smoothing=0.0):
    if class_counts is not None:
        # Inverse frequency weighting
        total = sum(class_counts)
        weights = [total / c if c > 0 else 0 for c in class_counts]
        weights = torch.FloatTensor(weights)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
