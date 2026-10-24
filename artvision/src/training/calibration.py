import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, is_binary: bool = False):
    """
    Computes Expected Calibration Error (ECE) for multi-class or binary predictions.
    """
    if is_binary:
        # Binary: probs is 1D probability of class 1
        confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
        predictions = (probs >= 0.5).astype(int)
        accuracies = (predictions == labels).astype(float)
    else:
        # Multi-class
        confidences = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)
        accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        bin_counts.append(int(np.sum(in_bin)))
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return float(ece), bin_boundaries, bin_accs, bin_confs, bin_counts


class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling fitted per head on the validation set.
    """
    def __init__(self):
        super().__init__()
        self.temperatures = {}

    def fit_head(self, logits: torch.Tensor, targets: torch.Tensor, task_type: str = "classification", lr: float = 0.01, max_iter: int = 200):
        """
        Fits optimal temperature T > 0 on validation logits and targets using L-BFGS or Adam.
        """
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        if task_type == "binary":
            criterion = nn.BCEWithLogitsLoss()
            targets_float = targets.float()

            def eval_loss():
                optimizer.zero_grad()
                scaled = logits / temperature.clamp(min=0.01)
                loss = criterion(scaled, targets_float)
                loss.backward()
                return loss
        else:
            criterion = nn.CrossEntropyLoss()

            def eval_loss():
                optimizer.zero_grad()
                scaled = logits / temperature.clamp(min=0.01)
                loss = criterion(scaled, targets)
                loss.backward()
                return loss

        optimizer.step(eval_loss)
        return float(temperature.item())

    def fit_all(self, val_logits_dict: dict, val_targets_dict: dict, val_masks_dict: dict, tasks_config: dict):
        for task_name, task_info in tasks_config.items():
            if not task_info.get("enabled", False) or task_name not in val_logits_dict:
                continue

            msk = val_masks_dict[task_name].bool()
            if msk.sum() == 0:
                self.temperatures[task_name] = 1.0
                continue

            logits = val_logits_dict[task_name][msk]
            targets = val_targets_dict[task_name][msk]
            task_type = task_info.get("type", "classification")

            temp = self.fit_head(logits, targets, task_type=task_type)
            self.temperatures[task_name] = max(0.05, temp)
            print(f"Fitted Temperature for '{task_name}': {self.temperatures[task_name]:.4f}")

    def scale_logits(self, logits: torch.Tensor, task_name: str) -> torch.Tensor:
        T = self.temperatures.get(task_name, 1.0)
        return logits / T

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.temperatures, f, indent=4)
        print(f"Saved calibration parameters to {filepath}")

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            self.temperatures = json.load(f)
        print(f"Loaded calibration parameters: {self.temperatures}")


def plot_reliability_diagram(bin_boundaries, bin_accs, ece: float, title: str, save_path: str):
    """Plots and saves a reliability diagram."""
    plt.figure(figsize=(6, 6))
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    plt.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, label="Outputs", color="royalblue", edgecolor="black")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"{title} (ECE: {ece*100:.2f}%)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
