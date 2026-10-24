import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import copy
import json
import yaml
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data.multitask_dataset import ArtVisionMultiTaskDataset
from src.models.multitask import MultiTaskModel
from src.training.multitask_loss import MaskedMultiTaskLoss
from src.training.sampler import BalancedMultiTaskBatchSampler
from src.training.seed import seed_everything


def evaluate(model, dataloader, criterion, device, tasks_config):
    model.eval()
    total_loss = 0.0
    task_losses_sum = {t: 0.0 for t in tasks_config if tasks_config[t].get("enabled", False)}

    all_preds = {t: [] for t in tasks_config if tasks_config[t].get("enabled", False)}
    all_targets = {t: [] for t in tasks_config if tasks_config[t].get("enabled", False)}
    all_probs = {t: [] for t in tasks_config if tasks_config[t].get("enabled", False)}
    all_logits = {t: [] for t in tasks_config if tasks_config[t].get("enabled", False)}
    all_masks = {t: [] for t in tasks_config if tasks_config[t].get("enabled", False)}

    with torch.no_grad():
        for images, targets, masks in dataloader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            masks = {k: v.to(device) for k, v in masks.items()}

            outputs = model(images)
            loss, task_losses, _ = criterion(outputs, targets, masks)

            total_loss += loss.item()
            for t, l_val in task_losses.items():
                task_losses_sum[t] += l_val.item()

            for t in all_preds:
                msk = masks[t].bool().cpu()
                all_masks[t].append(msk)
                out = outputs[t].cpu()
                all_logits[t].append(out)

                if t == "authenticity":
                    probs = torch.sigmoid(out)
                    preds = (probs > 0.5).long()
                    all_probs[t].append(probs)
                    all_preds[t].append(preds)
                    all_targets[t].append(targets[t].cpu())
                else:
                    probs = torch.softmax(out, dim=-1)
                    preds = out.argmax(dim=-1)
                    all_probs[t].append(probs)
                    all_preds[t].append(preds)
                    all_targets[t].append(targets[t].cpu())

    metrics = {}
    macro_f1s = []

    for t in all_preds:
        masks_cat = torch.cat(all_masks[t]).numpy()
        if masks_cat.sum() == 0:
            metrics[f"{t}_macro_f1"] = 0.0
            metrics[f"{t}_accuracy"] = 0.0
            continue

        preds_cat = torch.cat(all_preds[t]).numpy()[masks_cat]
        targets_cat = torch.cat(all_targets[t]).numpy()[masks_cat]
        probs_cat = torch.cat(all_probs[t]).numpy()[masks_cat]

        acc = accuracy_score(targets_cat, preds_cat)
        f1 = f1_score(targets_cat, preds_cat, average="macro", zero_division=0)
        metrics[f"{t}_accuracy"] = float(acc)
        metrics[f"{t}_macro_f1"] = float(f1)
        macro_f1s.append(f1)

        if t == "authenticity" and len(np.unique(targets_cat)) > 1:
            try:
                metrics[f"{t}_auroc"] = float(roc_auc_score(targets_cat, probs_cat))
            except Exception:
                pass

    metrics["mean_macro_f1"] = float(np.mean(macro_f1s)) if macro_f1s else 0.0
    metrics["val_loss"] = total_loss / max(1, len(dataloader))

    # Package raw logits and targets for calibration
    raw_eval_data = {
        "logits": {t: torch.cat(all_logits[t]) for t in all_logits},
        "targets": {t: torch.cat(all_targets[t]) for t in all_targets},
        "masks": {t: torch.cat(all_masks[t]) for t in all_masks},
    }

    return metrics, raw_eval_data


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    task_losses_sum = {}
    weights_sum = {}

    for images, targets, masks in dataloader:
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        masks = {k: v.to(device) for k, v in masks.items()}

        optimizer.zero_grad()
        outputs = model(images)
        loss, task_losses, active_weights = criterion(outputs, targets, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        for t, l_val in task_losses.items():
            task_losses_sum[t] = task_losses_sum.get(t, 0.0) + l_val.item()
        for t, w_val in active_weights.items():
            weights_sum[t] = weights_sum.get(t, 0.0) + float(w_val)

    avg_loss = total_loss / max(1, len(dataloader))
    avg_task_losses = {t: val / max(1, len(dataloader)) for t, val in task_losses_sum.items()}
    avg_weights = {t: val / max(1, len(dataloader)) for t, val in weights_sum.items()}
    return avg_loss, avg_task_losses, avg_weights


def train_multitask(config_path: str, enabled_tasks_list=None, weighting_method=None, dev_run: bool = False, output_dir=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if enabled_tasks_list:
        for t in config["tasks"]:
            config["tasks"][t]["enabled"] = (t in enabled_tasks_list)

    if weighting_method:
        config["loss_weighting"]["method"] = weighting_method

    if output_dir:
        config["training"]["output_dir"] = output_dir

    seed_everything(config["training"]["seed"])
    device = torch.device(config["training"]["device"] if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"Training Multi-Task Model on device: {device}")
    print(f"Enabled Tasks: {[t for t, info in config['tasks'].items() if info.get('enabled', False)]}")
    print(f"Loss Weighting: {config['loss_weighting']['method']}")
    print(f"Output Directory: {config['training']['output_dir']}")
    print(f"=======================================================\n")

    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(config["data"]["image_size"], scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    train_ds = ArtVisionMultiTaskDataset(
        metadata_path=config["data"]["metadata_path"],
        image_dir=config["data"]["image_dir"],
        split=config["data"]["train_split"],
        transform=transform_train,
        artist_top_n=config["tasks"].get("artist", {}).get("top_n", 50),
        artist_min_samples=config["tasks"].get("artist", {}).get("min_samples", 40),
    )

    val_ds = ArtVisionMultiTaskDataset(
        metadata_path=config["data"]["metadata_path"],
        image_dir=config["data"]["image_dir"],
        split=config["data"]["val_split"],
        transform=transform_eval,
        artist_top_n=config["tasks"].get("artist", {}).get("top_n", 50),
        artist_min_samples=config["tasks"].get("artist", {}).get("min_samples", 40),
    )

    # Adjust config num_classes dynamically
    config["tasks"]["style"]["num_classes"] = len(train_ds.style_classes)
    config["tasks"]["artist"]["top_n"] = len(train_ds.artist_classes)
    config["tasks"]["mood"]["num_classes"] = len(train_ds.mood_classes)
    if "genre" in config["tasks"]:
        config["tasks"]["genre"]["num_classes"] = len(train_ds.genre_classes)

    # Class imbalance weights
    class_weights = {}
    for t in config["tasks"]:
        if config["tasks"][t].get("enabled", False):
            counts = train_ds.get_class_counts(t)
            if counts and sum(counts) > 0:
                tot = sum(counts)
                weights = [tot / max(1, c) if c > 0 else 0.0 for c in counts]
                # Normalize so mean weight is 1.0
                mean_w = sum(weights) / max(1, len([w for w in weights if w > 0]))
                weights = [w / max(1e-4, mean_w) for w in weights]
                class_weights[t] = torch.tensor(weights, dtype=torch.float32)

    # Sampler for train
    sampler_cfg = config.get("sampler", {})
    if sampler_cfg.get("guarantee_minimums", True) and not dev_run:
        train_sampler = BalancedMultiTaskBatchSampler(
            dataset=train_ds,
            batch_size=config["data"]["batch_size"],
            min_samples_per_task=sampler_cfg.get("min_samples_per_task", {}),
            seed=config["training"]["seed"]
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=config["data"]["num_workers"]
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=config["data"]["batch_size"],
            shuffle=True,
            num_workers=config["data"]["num_workers"]
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"]
    )

    # Model
    model = MultiTaskModel(config).to(device)

    # Loss
    manual_w = {t: config["tasks"][t].get("weight", 1.0) for t in config["tasks"]}
    criterion = MaskedMultiTaskLoss(
        tasks_config=config["tasks"],
        class_weights=class_weights,
        weighting_method=config["loss_weighting"]["method"],
        manual_weights=manual_w
    ).to(device)

    history = []
    best_mean_f1 = -1.0
    patience_counter = 0

    epochs_stage1 = 2 if dev_run else config["training"]["epochs_stage1"]
    epochs_stage2 = 2 if dev_run else config["training"]["epochs_stage2"]

    # ================= Stage 1: Warmup Heads =================
    print("--- Stage 1: Training Heads Only (Backbone Frozen) ---")
    model.freeze_backbone()
    
    # Optimizer for heads and loss params
    head_params = list(model.heads.parameters())
    loss_params = list(criterion.parameters())
    optimizer = torch.optim.AdamW(head_params + loss_params, lr=config["training"]["lr_head"], weight_decay=config["training"]["weight_decay"])

    for epoch in range(1, epochs_stage1 + 1):
        t0 = time.time()
        tr_loss, tr_task_losses, tr_weights = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_metrics, _ = evaluate(model, val_loader, criterion, device, config["tasks"])
        el = time.time() - t0

        print(f"Stage 1 Epoch {epoch:02d}/{epochs_stage1:02d} [{el:.1f}s] - Train Loss: {tr_loss:.4f} - Val Loss: {v_metrics['val_loss']:.4f} - Mean Val Macro-F1: {v_metrics['mean_macro_f1']:.4f}")
        for t in model.enabled_tasks:
            acc = v_metrics.get(f"{t}_accuracy", 0.0)
            f1 = v_metrics.get(f"{t}_macro_f1", 0.0)
            print(f"   -> {t.capitalize():<14} Val Acc: {acc:.4f} | Val F1: {f1:.4f} | Loss: {tr_task_losses.get(t, 0.0):.4f} | Weight: {tr_weights.get(t, 1.0):.4f}")

        history.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": tr_loss,
            "task_losses": tr_task_losses,
            "weights": tr_weights,
            "val_metrics": v_metrics
        })

    # ================= Stage 2: Full Fine-tuning =================
    print("\n--- Stage 2: Fine-Tuning Full Model (Backbone Unfrozen) ---")
    model.unfreeze_backbone()

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["training"]["lr_backbone"]},
        {"params": model.heads.parameters(), "lr": config["training"]["lr_head"] * 0.5},
        {"params": criterion.parameters(), "lr": config["training"]["lr_head"] * 0.5},
    ], weight_decay=config["training"]["weight_decay"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)

    best_checkpoint_path = os.path.join(config["training"]["output_dir"], "best_model.pth")

    for epoch in range(1, epochs_stage2 + 1):
        t0 = time.time()
        tr_loss, tr_task_losses, tr_weights = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_metrics, raw_val_data = evaluate(model, val_loader, criterion, device, config["tasks"])
        scheduler.step()
        el = time.time() - t0

        mean_f1 = v_metrics["mean_macro_f1"]
        print(f"Stage 2 Epoch {epoch:02d}/{epochs_stage2:02d} [{el:.1f}s] - Train Loss: {tr_loss:.4f} - Val Loss: {v_metrics['val_loss']:.4f} - Mean Val Macro-F1: {mean_f1:.4f}")
        for t in model.enabled_tasks:
            acc = v_metrics.get(f"{t}_accuracy", 0.0)
            f1 = v_metrics.get(f"{t}_macro_f1", 0.0)
            print(f"   -> {t.capitalize():<14} Val Acc: {acc:.4f} | Val F1: {f1:.4f} | Loss: {tr_task_losses.get(t, 0.0):.4f} | Weight: {tr_weights.get(t, 1.0):.4f}")

        history.append({
            "stage": 2,
            "epoch": epoch,
            "train_loss": tr_loss,
            "task_losses": tr_task_losses,
            "weights": tr_weights,
            "val_metrics": v_metrics
        })

        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "config": config,
                "best_mean_f1": best_mean_f1,
                "epoch": epoch,
                "artist_classes": train_ds.artist_classes,
                "style_classes": train_ds.style_classes,
                "mood_classes": train_ds.mood_classes,
                "genre_classes": train_ds.genre_classes,
            }, best_checkpoint_path)
            print(f"   [*] Saved new best model checkpoint! (Mean Val F1: {best_mean_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    with open(os.path.join(config["training"]["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    print(f"\nTraining completed! Best Mean Val Macro-F1: {best_mean_f1:.4f}")
    return best_checkpoint_path, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task artwork analysis training")
    parser.add_argument("--config", type=str, default="configs/train_multitask.yaml")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated tasks to enable, e.g. style,genre")
    parser.add_argument("--weighting", type=str, default=None, help="Loss weighting: equal, manual, uncertainty")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    enabled_tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    train_multitask(
        config_path=args.config,
        enabled_tasks_list=enabled_tasks,
        weighting_method=args.weighting,
        dev_run=args.dev,
        output_dir=args.output_dir
    )
