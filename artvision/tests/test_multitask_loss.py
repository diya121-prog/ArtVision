import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from src.training.multitask_loss import MaskedMultiTaskLoss


def test_all_masked_task_returns_zero_no_nan():
    tasks_cfg = {
        "style": {"enabled": True, "type": "classification"},
        "genre": {"enabled": True, "type": "classification"},
        "authenticity": {"enabled": True, "type": "binary"},
    }
    loss_fn = MaskedMultiTaskLoss(tasks_cfg, weighting_method="equal")

    B = 8
    predictions = {
        "style": torch.randn(B, 10, requires_grad=True),
        "genre": torch.randn(B, 10, requires_grad=True),
        "authenticity": torch.randn(B, requires_grad=True),
    }
    targets = {
        "style": torch.randint(0, 10, (B,)),
        "genre": torch.randint(0, 10, (B,)),
        "authenticity": torch.randint(0, 2, (B,)).float(),
    }
    # genre has all zeros in masks!
    masks = {
        "style": torch.ones(B),
        "genre": torch.zeros(B),
        "authenticity": torch.ones(B),
    }

    total_loss, task_losses, weights = loss_fn(predictions, targets, masks)
    
    assert not torch.isnan(total_loss), "Total loss is NaN!"
    assert task_losses["genre"].item() == 0.0, "All-masked genre loss must be 0.0"
    assert not torch.isnan(task_losses["genre"]), "Genre loss is NaN!"
    assert task_losses["style"].item() > 0.0, "Style loss should be positive"
    assert task_losses["authenticity"].item() > 0.0, "Authenticity loss should be positive"

    # Backward pass should work cleanly
    total_loss.backward()
    assert predictions["style"].grad is not None
    assert predictions["authenticity"].grad is not None


def test_mixed_mask_averaging():
    tasks_cfg = {"style": {"enabled": True, "type": "classification"}}
    loss_fn = MaskedMultiTaskLoss(tasks_cfg, weighting_method="equal")

    B = 4
    predictions = {"style": torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])}
    targets = {"style": torch.tensor([0, 0, 0, 0])}  # Samples 1 and 3 are wrong, 0 and 2 are right
    # Only evaluate samples 0 and 2 (both right)
    masks = {"style": torch.tensor([1.0, 0.0, 1.0, 0.0])}

    total_loss, task_losses, _ = loss_fn(predictions, targets, masks)
    # Since samples 0 and 2 have target 0 and large logit for 0, loss should be very small (~0)
    assert task_losses["style"].item() < 0.01, f"Loss was {task_losses['style'].item()}, expected near 0"


def test_uncertainty_weighting_and_regularizer():
    tasks_cfg = {
        "style": {"enabled": True, "type": "classification"},
        "mood": {"enabled": True, "type": "classification", "loss_mode": "kl"},
    }
    loss_fn = MaskedMultiTaskLoss(tasks_cfg, weighting_method="uncertainty")

    B = 4
    predictions = {
        "style": torch.randn(B, 10, requires_grad=True),
        "mood": torch.randn(B, 5, requires_grad=True),
    }
    targets = {
        "style": torch.randint(0, 10, (B,)),
        "mood": torch.randint(0, 5, (B,)),
        "mood_dist": torch.softmax(torch.randn(B, 5), dim=-1),
    }
    masks = {
        "style": torch.ones(B),
        "mood": torch.ones(B),
    }

    total_loss, task_losses, weights = loss_fn(predictions, targets, masks)
    assert not torch.isnan(total_loss)
    assert "style" in weights and "mood" in weights
    
    # Check backward to learnable log_vars
    total_loss.backward()
    assert loss_fn.log_vars["style"].grad is not None
    assert loss_fn.log_vars["mood"].grad is not None


if __name__ == "__main__":
    test_all_masked_task_returns_zero_no_nan()
    test_mixed_mask_averaging()
    test_uncertainty_weighting_and_regularizer()
    print("All masked loss tests passed!")
