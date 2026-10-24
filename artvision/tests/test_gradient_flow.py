import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import torch
from src.models.multitask import MultiTaskModel


def test_gradient_reach_backbone():
    with open("configs/train_multitask.yaml") as f:
        config = yaml.safe_load(f)

    # Disable pretrained download for quick unit test
    config["model"]["pretrained"] = False
    config["model"]["pretrained_checkpoint"] = None
    model = MultiTaskModel(config)
    model.unfreeze_backbone()

    x = torch.randn(2, 3, 224, 224)

    # For each enabled head, compute loss and verify gradient in backbone
    for task_name in model.enabled_tasks:
        model.zero_grad()
        outputs = model(x)
        out = outputs[task_name]
        
        # Simple scalar loss
        loss = out.sum()
        loss.backward()
        
        # Check first conv layer gradient
        stem_grad = model.backbone.backbone.conv_stem.weight.grad
        assert stem_grad is not None, f"Gradient did not reach conv_stem from task {task_name}!"
        assert stem_grad.abs().sum() > 0, f"Gradient in conv_stem is zero from task {task_name}!"
        print(f"Gradient verified from head '{task_name}' to backbone conv_stem.")

    print("Gradient reach test passed for all enabled heads!")


if __name__ == "__main__":
    test_gradient_reach_backbone()
