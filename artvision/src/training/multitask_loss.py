import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMultiTaskLoss(nn.Module):
    """
    Masked multi-task loss module supporting:
    - Independent sample masking per task (0 loss and no NaN if 0 labeled samples)
    - Loss averaging over task's own labeled samples
    - Class-imbalance weighting
    - Hard CE vs Soft KL divergence for mood
    - Weighting schemes: 'equal', 'manual', 'uncertainty' (Kendall et al.)
    """
    def __init__(
        self,
        tasks_config: dict,
        class_weights: dict = None,
        weighting_method: str = "uncertainty",
        manual_weights: dict = None,
    ):
        super().__init__()
        self.tasks_config = tasks_config
        self.weighting_method = weighting_method.lower()
        self.manual_weights = manual_weights or {}
        
        # Loss criteria per task
        self.class_weights = class_weights or {}
        
        # Uncertainty weighting learnable parameters: log(sigma^2)
        # Kendall et al. Multi-Task Learning Using Uncertainty to Weigh Losses
        self.log_vars = nn.ParameterDict()
        for task_name, task_info in tasks_config.items():
            if task_info.get("enabled", False):
                self.log_vars[task_name] = nn.Parameter(torch.zeros(1))

    def forward(self, predictions: dict, targets: dict, masks: dict):
        task_losses = {}
        active_weights = {}

        for task_name, task_info in self.tasks_config.items():
            if not task_info.get("enabled", False) or task_name not in predictions:
                continue

            preds = predictions[task_name]
            tgts = targets[task_name]
            msk = masks[task_name]

            num_labeled = msk.sum()
            if num_labeled == 0:
                # Safe zero loss with gradient graph intact (no backprop impact)
                task_losses[task_name] = torch.tensor(0.0, device=preds.device, dtype=torch.float32)
                continue

            # Compute per-sample loss
            if task_name == "authenticity":
                # Binary classification
                weights = self.class_weights.get(task_name, None)
                pos_weight = weights[1] / weights[0] if weights is not None and len(weights) == 2 and weights[0] > 0 else None
                if pos_weight is not None:
                    pos_weight = torch.tensor([pos_weight], device=preds.device)
                bce = F.binary_cross_entropy_with_logits(preds, tgts.float(), pos_weight=pos_weight, reduction="none")
                task_loss = (bce * msk).sum() / num_labeled

            elif task_name == "mood" and task_info.get("loss_mode", "kl") == "kl" and "mood_dist" in targets:
                # Soft-label KL Divergence
                mood_dist = targets["mood_dist"].to(preds.device)
                log_probs = F.log_softmax(preds, dim=-1)
                # KL per sample: sum_c target_c * (log(target_c) - log_pred_c)
                # PyTorch KLDivLoss(log_probs, target) computes target * (log(target) - log_probs)
                kl_per_sample = F.kl_div(log_probs, mood_dist, reduction="none").sum(dim=-1)
                task_loss = (kl_per_sample * msk).sum() / num_labeled

            else:
                # Standard Softmax Cross-Entropy
                weights = self.class_weights.get(task_name, None)
                if weights is not None:
                    weights = weights.to(preds.device)
                ce = F.cross_entropy(preds, tgts, weight=weights, reduction="none")
                task_loss = (ce * msk).sum() / num_labeled

            task_losses[task_name] = task_loss

        # Combine task losses according to chosen weighting scheme
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device, dtype=torch.float32)

        for task_name, loss_val in task_losses.items():
            msk = masks[task_name]
            num_labeled = msk.sum()
            
            if self.weighting_method == "equal":
                weight = 1.0
                active_weights[task_name] = weight
                total_loss = total_loss + weight * loss_val

            elif self.weighting_method == "manual":
                weight = float(self.manual_weights.get(task_name, 1.0))
                active_weights[task_name] = weight
                total_loss = total_loss + weight * loss_val

            elif self.weighting_method == "uncertainty":
                s = self.log_vars[task_name]
                # Learned weight: exp(-s)
                effective_weight = torch.exp(-s).item()
                active_weights[task_name] = effective_weight
                
                # Only include Kendall uncertainty regularizer if this batch has labeled samples for the task
                if num_labeled > 0:
                    total_loss = total_loss + 0.5 * torch.exp(-s) * loss_val + 0.5 * s
                else:
                    total_loss = total_loss + 0.0 * s
            else:
                total_loss = total_loss + loss_val
                active_weights[task_name] = 1.0

        return total_loss, task_losses, active_weights

    def get_current_task_weights(self) -> dict:
        """Returns the current effective loss weight for each task."""
        weights = {}
        for task_name in self.tasks_config:
            if not self.tasks_config[task_name].get("enabled", False):
                continue
            if self.weighting_method == "uncertainty" and task_name in self.log_vars:
                s = self.log_vars[task_name].item()
                weights[task_name] = float(math.exp(-s))
            elif self.weighting_method == "manual":
                weights[task_name] = float(self.manual_weights.get(task_name, 1.0))
            else:
                weights[task_name] = 1.0
        return weights
