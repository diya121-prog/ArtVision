import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score

from src.data.multitask_dataset import ArtVisionMultiTaskDataset, STYLE_CLASSES, MOOD_CLASSES
from src.models.multitask import MultiTaskModel
from src.training.calibration import TemperatureScaler, calculate_ece, plot_reliability_diagram


def evaluate_multitask_model(checkpoint_path: str, output_dir: str = None, split: str = "test"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]

    if output_dir is None:
        output_dir = os.path.join(config["training"]["output_dir"], "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(config["training"]["device"] if torch.backends.mps.is_available() else "cpu")
    print(f"Evaluating checkpoint '{checkpoint_path}' on {split} set (Device: {device})...")

    # Load label maps
    with open("configs/label_maps.yaml") as f:
        label_maps = yaml.safe_load(f)
    movement_map = label_maps.get("movement_map", {})

    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = ArtVisionMultiTaskDataset(
        metadata_path=config["data"]["metadata_path"],
        image_dir=config["data"]["image_dir"],
        split=split,
        transform=transform_eval,
        artist_top_n=config["tasks"].get("artist", {}).get("top_n", 50),
        artist_min_samples=config["tasks"].get("artist", {}).get("min_samples", 40),
    )

    test_loader = DataLoader(test_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=0)

    # Initialize model
    model = MultiTaskModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load or fit temperature scaler on validation set
    calib_file = os.path.join(config["training"]["output_dir"], "calibration.json")
    scaler = TemperatureScaler()
    if os.path.exists(calib_file):
        scaler.load(calib_file)
    else:
        # Fit on val set
        print("Fitting temperature scaler on validation set...")
        val_ds = ArtVisionMultiTaskDataset(
            metadata_path=config["data"]["metadata_path"],
            image_dir=config["data"]["image_dir"],
            split="val",
            transform=transform_eval,
            artist_top_n=config["tasks"].get("artist", {}).get("top_n", 50),
            artist_min_samples=config["tasks"].get("artist", {}).get("min_samples", 40),
        )
        val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=0)
        val_logits = {t: [] for t in model.enabled_tasks}
        val_targets = {t: [] for t in model.enabled_tasks}
        val_masks = {t: [] for t in model.enabled_tasks}

        with torch.no_grad():
            for images, targets, masks in val_loader:
                images = images.to(device)
                outs = model(images)
                for t in model.enabled_tasks:
                    val_logits[t].append(outs[t].cpu())
                    val_targets[t].append(targets[t].cpu())
                    val_masks[t].append(masks[t].cpu())

        val_logits_cat = {t: torch.cat(val_logits[t]) for t in val_logits}
        val_targets_cat = {t: torch.cat(val_targets[t]) for t in val_targets}
        val_masks_cat = {t: torch.cat(val_masks[t]) for t in val_masks}
        scaler.fit_all(val_logits_cat, val_targets_cat, val_masks_cat, config["tasks"])
        os.makedirs(os.path.dirname(calib_file), exist_ok=True)
        scaler.save(calib_file)

    # Evaluation on Test set
    all_logits = {t: [] for t in model.enabled_tasks}
    all_targets = {t: [] for t in model.enabled_tasks}
    all_masks = {t: [] for t in model.enabled_tasks}
    raw_images = []

    inv_trans = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
    ])

    with torch.no_grad():
        for images, targets, masks in test_loader:
            images = images.to(device)
            outs = model(images)
            for t in model.enabled_tasks:
                all_logits[t].append(outs[t].cpu())
                all_targets[t].append(targets[t].cpu())
                all_masks[t].append(masks[t].cpu())
            for img in images.cpu():
                raw_images.append(inv_trans(img).permute(1, 2, 0).clamp(0, 1).numpy())

    all_logits = {t: torch.cat(all_logits[t]) for t in all_logits}
    all_targets = {t: torch.cat(all_targets[t]) for t in all_targets}
    all_masks = {t: torch.cat(all_masks[t]) for t in all_masks}

    task_results = {}

    for t in model.enabled_tasks:
        msk = all_masks[t].bool().numpy()
        num_eval = int(msk.sum())
        if num_eval == 0:
            print(f"Task '{t}': 0 test samples with valid mask. Skipping metrics.")
            continue

        logits = all_logits[t][msk]
        targets = all_targets[t][msk].numpy()
        T = scaler.temperatures.get(t, 1.0)
        calibrated_logits = logits / T

        if t == "authenticity":
            probs_uncal = torch.sigmoid(logits).numpy()
            probs_cal = torch.sigmoid(calibrated_logits).numpy()
            preds_cal = (probs_cal >= 0.5).astype(int)

            acc = float(accuracy_score(targets, preds_cal))
            f1 = float(f1_score(targets, preds_cal, average="macro", zero_division=0))
            auroc = float(roc_auc_score(targets, probs_cal)) if len(np.unique(targets)) > 1 else 0.0

            ece_uncal, boundaries, accs, confs, _ = calculate_ece(probs_uncal, targets, is_binary=True)
            ece_cal, b_cal, a_cal, c_cal, _ = calculate_ece(probs_cal, targets, is_binary=True)

            plot_reliability_diagram(boundaries, accs, ece_uncal, f"{t.capitalize()} (Uncalibrated)", os.path.join(output_dir, f"calibration_{t}_uncalibrated.png"))
            plot_reliability_diagram(b_cal, a_cal, ece_cal, f"{t.capitalize()} (Calibrated, T={T:.2f})", os.path.join(output_dir, f"calibration_{t}_calibrated.png"))

            cm = confusion_matrix(targets, preds_cal)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
            plt.title(f"Confusion Matrix - {t.capitalize()}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_{t}.png"))
            plt.close()

            task_results[t] = {
                "num_samples": num_eval,
                "accuracy": acc,
                "macro_f1": f1,
                "auroc": auroc,
                "ece_uncalibrated": ece_uncal,
                "ece_calibrated": ece_cal,
                "temperature": T
            }
        else:
            probs_uncal = F.softmax(logits, dim=-1).numpy()
            probs_cal = F.softmax(calibrated_logits, dim=-1).numpy()
            preds_cal = np.argmax(probs_cal, axis=-1)

            acc = float(accuracy_score(targets, preds_cal))
            f1 = float(f1_score(targets, preds_cal, average="macro", zero_division=0))

            # Top-3 Accuracy
            top3_acc = 0.0
            if probs_cal.shape[1] >= 3:
                top3_preds = np.argsort(probs_cal, axis=1)[:, -3:]
                correct_top3 = sum([1 for i in range(len(targets)) if targets[i] in top3_preds[i]])
                top3_acc = float(correct_top3 / len(targets))

            ece_uncal, boundaries, accs, confs, _ = calculate_ece(probs_uncal, targets, is_binary=False)
            ece_cal, b_cal, a_cal, c_cal, _ = calculate_ece(probs_cal, targets, is_binary=False)

            plot_reliability_diagram(boundaries, accs, ece_uncal, f"{t.capitalize()} (Uncalibrated)", os.path.join(output_dir, f"calibration_{t}_uncalibrated.png"))
            plot_reliability_diagram(b_cal, a_cal, ece_cal, f"{t.capitalize()} (Calibrated, T={T:.2f})", os.path.join(output_dir, f"calibration_{t}_calibrated.png"))

            # Class names
            if t == "style":
                classes = test_ds.style_classes
            elif t == "artist":
                classes = test_ds.artist_classes
            elif t == "genre":
                classes = test_ds.genre_classes
            elif t == "mood":
                classes = test_ds.mood_classes
            else:
                classes = [str(i) for i in range(probs_cal.shape[1])]

            # Confusion Matrix
            cm = confusion_matrix(targets, preds_cal, labels=list(range(len(classes))))
            plt.figure(figsize=(max(6, len(classes)*0.6), max(5, len(classes)*0.5)))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            plt.ylabel("True")
            plt.xlabel("Predicted")
            plt.title(f"Confusion Matrix - {t.capitalize()}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"confusion_matrix_{t}.png"))
            plt.close()

            # Classification report table
            try:
                rep = classification_report(targets, preds_cal, target_names=classes, output_dict=True, zero_division=0)
                pd.DataFrame(rep).transpose().to_csv(os.path.join(output_dir, f"classification_report_{t}.csv"))
            except Exception:
                pass

            task_results[t] = {
                "num_samples": num_eval,
                "accuracy": acc,
                "top3_accuracy": top3_acc if probs_cal.shape[1] >= 3 else None,
                "macro_f1": f1,
                "ece_uncalibrated": ece_uncal,
                "ece_calibrated": ece_cal,
                "temperature": T
            }

            if t == "mood":
                task_results[t]["experimental"] = True
                # 1000-iteration bootstrap CI for mood
                rng_boot = np.random.RandomState(42)
                boot_accs, boot_f1s = [], []
                n_samples = len(targets)
                for _ in range(1000):
                    b_idx = rng_boot.choice(n_samples, size=n_samples, replace=True)
                    b_true = targets[b_idx]
                    b_pred = preds_cal[b_idx]
                    boot_accs.append(accuracy_score(b_true, b_pred))
                    boot_f1s.append(f1_score(b_true, b_pred, average="macro", zero_division=0))
                task_results[t]["bootstrap_accuracy_95ci"] = [float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))]
                task_results[t]["bootstrap_macro_f1_95ci"] = [float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5))]

            # Derived Movement accuracy (from style)
            if t == "style":
                true_styles = [classes[idx] for idx in targets]
                pred_styles = [classes[idx] for idx in preds_cal]
                true_movements = [movement_map.get(s, "Unknown") for s in true_styles]
                pred_movements = [movement_map.get(s, "Unknown") for s in pred_styles]
                movement_acc = float(np.mean([t_m == p_m for t_m, p_m in zip(true_movements, pred_movements)]))
                task_results["derived_movement"] = {
                    "accuracy": movement_acc,
                    "num_samples": len(true_movements)
                }

        # Misclassifications Grid (8-12 samples)
        wrong_indices = np.where(targets != preds_cal)[0]
        if len(wrong_indices) > 0:
            num_show = min(12, len(wrong_indices))
            fig, axes = plt.subplots(3, 4, figsize=(14, 11))
            # Find the actual indices in test_ds
            eval_to_orig_idx = np.where(msk)[0]

            for i, ax in enumerate(axes.flat):
                if i < num_show:
                    w_i = wrong_indices[i]
                    orig_idx = eval_to_orig_idx[w_i]
                    img = raw_images[orig_idx]
                    ax.imshow(img)

                    if t == "authenticity":
                        t_lbl = "AI" if targets[w_i] == 1 else "Human"
                        p_lbl = "AI" if preds_cal[w_i] == 1 else "Human"
                        conf = probs_cal[w_i] if preds_cal[w_i] == 1 else (1.0 - probs_cal[w_i])
                    else:
                        t_lbl = classes[targets[w_i]]
                        p_lbl = classes[preds_cal[w_i]]
                        conf = probs_cal[w_i, preds_cal[w_i]]

                    ax.set_title(f"True: {t_lbl}\nPred: {p_lbl}\nConf: {conf*100:.1f}%", fontsize=9, color="crimson")
                ax.axis("off")
            plt.suptitle(f"Misclassified Examples - {t.capitalize()}", fontsize=13)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"misclassifications_{t}.png"))
            plt.close()

    # Save metrics JSON
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(task_results, f, indent=4)

    print("\n--- Final Test Metrics Summary ---")
    print(json.dumps(task_results, indent=2))
    return task_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs/multitask_model/best_model.pth")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    evaluate_multitask_model(args.checkpoint, args.output_dir, args.split)
