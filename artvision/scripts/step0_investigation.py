import os
import sys
import json
import yaml
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.multitask_dataset import ArtVisionMultiTaskDataset, STYLE_CLASSES, MOOD_CLASSES
from src.models.multitask import MultiTaskModel
from src.training.calibration import TemperatureScaler

def run_step0():
    print("="*60)
    print("STEP 0a: STYLE CLASSES AND ALIASING ANALYSIS")
    print("="*60)
    
    metadata_path = "data/processed/metadata.csv"
    df = pd.read_csv(metadata_path)
    
    # Checkpoint and config
    ckpt = torch.load("runs/multitask_model/best_model.pth", map_location="cpu")
    config = ckpt["config"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Print all style class names with train counts
    train_df = df[df["split"] == "train"]
    style_train_counts = train_df["style"].value_counts(dropna=False)
    print("\n[Style Class Names with Train Counts in metadata.csv]")
    for s, count in style_train_counts.items():
        in_model = "YES (model class)" if s in STYLE_CLASSES else "NO (unmapped in model)"
        print(f"  - {s!r:<28}: count = {count:>5}  [{in_model}]")
    print(f"Total unique style names in train: {len(style_train_counts)}")
    print(f"Model output heads for style: {len(STYLE_CLASSES)}")
    
    # Evaluate baseline test predictions
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_ds = ArtVisionMultiTaskDataset(
        metadata_path=metadata_path,
        image_dir="data/processed/images",
        split="test",
        transform=transform_eval
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = MultiTaskModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    all_logits = {t: [] for t in model.enabled_tasks}
    all_targets = {t: [] for t in model.enabled_tasks}
    all_masks = {t: [] for t in model.enabled_tasks}
    
    with torch.no_grad():
        for images, targets, masks in test_loader:
            images = images.to(device)
            outs = model(images)
            for t in model.enabled_tasks:
                all_logits[t].append(outs[t].cpu())
                all_targets[t].append(targets[t].cpu())
                all_masks[t].append(masks[t].cpu())
                
    all_logits = {t: torch.cat(all_logits[t]) for t in all_logits}
    all_targets = {t: torch.cat(all_targets[t]) for t in all_targets}
    all_masks = {t: torch.cat(all_masks[t]) for t in all_masks}
    
    scaler = TemperatureScaler()
    scaler.load("runs/multitask_model/calibration.json")
    
    style_T = scaler.temperatures.get("style", 1.0)
    style_msk = all_masks["style"].bool().numpy()
    style_logits = all_logits["style"][style_msk]
    style_targets = all_targets["style"][style_msk].numpy()
    style_probs = F.softmax(style_logits / style_T, dim=-1).numpy()
    
    # Metrics before merging
    preds_before = np.argmax(style_probs, axis=-1)
    acc_before = float(accuracy_score(style_targets, preds_before))
    top3_preds_before = np.argsort(style_probs, axis=1)[:, -3:]
    top3_acc_before = float(sum([1 for i in range(len(style_targets)) if style_targets[i] in top3_preds_before[i]]) / len(style_targets))
    macro_f1_before = float(f1_score(style_targets, preds_before, average="macro", zero_division=0))
    
    print("\n--- Style Metrics on Test (BEFORE merging) ---")
    print(f"Num samples: {len(style_targets)}")
    print(f"Top-1 Accuracy: {acc_before*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc_before*100:.2f}%")
    print(f"Macro-F1:       {macro_f1_before*100:.2f}%")
    
    # Test with aliased / merged probabilities
    # 1. Alias map for raw names to canonical STYLE_CLASSES:
    # 2. Alias map for sub-styles (Analytical/Synthetic Cubism -> Cubism, Contemporary/New Realism -> Realism, Color Field / Action Painting -> Abstract Expressionism)
    alias_map_internal = {
        "Analytical_Cubism": "Cubism",
        "Synthetic_Cubism": "Cubism",
        "Contemporary_Realism": "Realism",
        "New_Realism": "Realism",
        "Action_painting": "Abstract_Expressionism",
        "Color_Field_Painting": "Abstract_Expressionism"
    }
    
    # Build merged class list
    merged_classes = []
    for c in STYLE_CLASSES:
        target_c = alias_map_internal.get(c, c)
        if target_c not in merged_classes:
            merged_classes.append(target_c)
            
    merged_probs = np.zeros((len(style_targets), len(merged_classes)))
    for orig_idx, c in enumerate(STYLE_CLASSES):
        target_c = alias_map_internal.get(c, c)
        new_idx = merged_classes.index(target_c)
        merged_probs[:, new_idx] += style_probs[:, orig_idx]
        
    merged_targets = np.array([
        merged_classes.index(alias_map_internal.get(STYLE_CLASSES[t], STYLE_CLASSES[t]))
        for t in style_targets
    ])
    
    preds_after = np.argmax(merged_probs, axis=-1)
    acc_after = float(accuracy_score(merged_targets, preds_after))
    top3_preds_after = np.argsort(merged_probs, axis=1)[:, -3:]
    top3_acc_after = float(sum([1 for i in range(len(merged_targets)) if merged_targets[i] in top3_preds_after[i]]) / len(merged_targets))
    macro_f1_after = float(f1_score(merged_targets, preds_after, average="macro", zero_division=0))
    
    print("\n--- Style Metrics on Test (AFTER merging sub-style variants into parents) ---")
    print(f"Merged classes count: {len(merged_classes)}")
    print(f"Top-1 Accuracy: {acc_after*100:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc_after*100:.2f}%")
    print(f"Macro-F1:       {macro_f1_after*100:.2f}%")

    print("\n" + "="*60)
    print("STEP 0b: AUTHENTICITY SHORTCUT DEEP DIVE")
    print("="*60)
    
    auth_T = scaler.temperatures.get("authenticity", 1.0)
    auth_logits = all_logits["authenticity"].squeeze(-1)
    auth_targets = all_targets["authenticity"].numpy()
    auth_probs = torch.sigmoid(auth_logits / auth_T).numpy()
    auth_preds = (auth_probs >= 0.5).astype(int)
    
    test_data = test_ds.data
    # Mask by source dataset
    is_ai_artbench = (test_data["source_dataset"] == "ai_artbench").values
    is_wikiart = (test_data["source_dataset"] == "huggan/wikiart").values
    is_wikiart_emotions = (test_data["source_dataset"] == "wikiart_emotions").values
    
    # 1. Test metrics separated
    print("\n1. Separated Authenticity Test Metrics:")
    
    # (i) AI-ArtBench only (100 real vs 100 AI)
    t_ai = auth_targets[is_ai_artbench]
    p_ai = auth_preds[is_ai_artbench]
    prob_ai = auth_probs[is_ai_artbench]
    cm_ai = confusion_matrix(t_ai, p_ai, labels=[0, 1])
    acc_ai = accuracy_score(t_ai, p_ai)
    f1_ai = f1_score(t_ai, p_ai, average="macro")
    auc_ai = roc_auc_score(t_ai, prob_ai)
    
    print(f"--- (i) AI-ArtBench Test Only (Total: {len(t_ai)}) ---")
    print(f"Accuracy: {acc_ai*100:.2f}%, Macro-F1: {f1_ai*100:.2f}%, AUROC: {auc_ai:.4f}")
    print("Confusion Matrix (Rows=True, Cols=Pred) [0=Human, 1=AI]:")
    print(f"  Human: TN={cm_ai[0,0]}, FP={cm_ai[0,1]}")
    print(f"  AI:    FN={cm_ai[1,0]}, TP={cm_ai[1,1]}")
    
    # (ii) WikiArt-Human (616 test) vs AI (100 test)
    # Human from wikiart + wikiart_emotions vs AI from ai_artbench
    is_all_human = (auth_targets == 0)
    is_ai = (auth_targets == 1)
    # Wikiart human (616) vs AI (100)
    is_wiki_vs_ai = (is_wikiart & is_all_human) | is_ai
    t_w_ai = auth_targets[is_wiki_vs_ai]
    p_w_ai = auth_preds[is_wiki_vs_ai]
    prob_w_ai = auth_probs[is_wiki_vs_ai]
    cm_w_ai = confusion_matrix(t_w_ai, p_w_ai, labels=[0, 1])
    acc_w_ai = accuracy_score(t_w_ai, p_w_ai)
    f1_w_ai = f1_score(t_w_ai, p_w_ai, average="macro")
    auc_w_ai = roc_auc_score(t_w_ai, prob_w_ai)
    
    print(f"\n--- (ii) WikiArt-Human ({sum(is_wikiart & is_all_human)}) vs AI ({sum(is_ai)}) (Total: {len(t_w_ai)}) ---")
    print(f"Accuracy: {acc_w_ai*100:.2f}%, Macro-F1: {f1_w_ai*100:.2f}%, AUROC: {auc_w_ai:.4f}")
    print("Confusion Matrix (Rows=True, Cols=Pred) [0=Human, 1=AI]:")
    print(f"  Human: TN={cm_w_ai[0,0]}, FP={cm_w_ai[0,1]}")
    print(f"  AI:    FN={cm_w_ai[1,0]}, TP={cm_w_ai[1,1]}")
    
    # 2. Always predict human baseline
    total_human = sum(auth_targets == 0)
    total_samples = len(auth_targets)
    always_human_acc = total_human / total_samples
    print(f"\n2. 'Always Predict Human' Baseline:")
    print(f"  {total_human} human / {total_samples} test = {always_human_acc*100:.2f}% accuracy (Macro-F1 = {f1_score(auth_targets, np.zeros_like(auth_targets), average='macro'):.4f})")
    
    # 3. Native image dimensions in data/raw/real-ai-art/
    print("\n3. Inspecting Native Image Dimensions in raw Real_AI_SD_LD_Dataset...")
    raw_dir = "data/raw/real-ai-art/Real_AI_SD_LD_Dataset/train"
    dims = {"real": [], "ai_sd": [], "ai_ld": []}
    
    for root, _, files in os.walk(raw_dir):
        category = None
        folder = os.path.basename(root)
        if folder.startswith("AI_SD"):
            category = "ai_sd"
        elif folder.startswith("AI_LD"):
            category = "ai_ld"
        elif any(folder.startswith(s) for s in ["art_nouveau", "baroque", "expressionism", "impressionism", "realism"]):
            category = "real"
            
        if category:
            for f in files[:20]: # sample 20 per folder
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    try:
                        with Image.open(os.path.join(root, f)) as im:
                            dims[category].append(im.size)
                    except Exception:
                        pass
                        
    for cat, sizes in dims.items():
        if sizes:
            unique_sizes = set(sizes)
            print(f"  Category '{cat}': sample {len(sizes)} images -> unique sizes: {list(unique_sizes)[:5]}")
            
    # 4. Laplacian variance / frequency energy & Logistic Regression on non-semantic features
    print("\n4. Non-Semantic Feature Extraction & Logistic Regression Baseline...")
    features_list = []
    targets_list = []
    
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        path = os.path.join("data/processed/images", f"{row['image_id']}.jpg")
        if not os.path.exists(path):
            path = os.path.join("data/processed/images", f"{row['image_id']}.png")
            
        try:
            with Image.open(path) as im:
                w, h = im.size
                aspect_ratio = float(w) / float(h)
                gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    continue
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                mean_val = float(np.mean(gray))
                std_val = float(np.std(gray))
                features_list.append([aspect_ratio, lap_var, mean_val, std_val])
                targets_list.append(int(row["is_ai_generated"]))
        except Exception:
            pass
            
    X = np.array(features_list)
    y = np.array(targets_list)
    
    lap_vars_human = X[y == 0, 1]
    lap_vars_ai = X[y == 1, 1]
    print(f"  Laplacian Variance (Sharpness):")
    print(f"    Human (n={len(lap_vars_human)}): mean = {np.mean(lap_vars_human):.2f}, std = {np.std(lap_vars_human):.2f}, median = {np.median(lap_vars_human):.2f}")
    print(f"    AI    (n={len(lap_vars_ai)}): mean = {np.mean(lap_vars_ai):.2f}, std = {np.std(lap_vars_ai):.2f}, median = {np.median(lap_vars_ai):.2f}")
    
    # Laplacian variance alone AUROC
    lap_auroc = roc_auc_score(y, X[:, 1])
    print(f"  AUROC of Laplacian Variance alone to predict AI: {lap_auroc:.4f}")
    
    # Logistic Regression on non-semantic features [aspect_ratio, lap_var, mean, std]
    # Standardize
    X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-7)
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(X_norm, y)
    lr_preds = lr.predict(X_norm)
    lr_probs = lr.predict_proba(X_norm)[:, 1]
    lr_acc = accuracy_score(y, lr_preds)
    lr_f1 = f1_score(y, lr_preds, average="macro")
    lr_auroc = roc_auc_score(y, lr_probs)
    print(f"  Logistic Regression on non-semantic features alone:")
    print(f"    Features: [aspect_ratio, laplacian_variance, gray_mean, gray_std]")
    print(f"    Accuracy: {lr_acc*100:.2f}%, Macro-F1: {lr_f1*100:.2f}%, AUROC: {lr_auroc:.4f}")
    print(f"    Coefficients: aspect_ratio={lr.coef_[0][0]:.3f}, laplacian_var={lr.coef_[0][1]:.3f}, mean={lr.coef_[0][2]:.3f}, std={lr.coef_[0][3]:.3f}")
    
    # 5. Robustness check: apply Gaussian blur (sigma=1.5) + downscale-upscale (0.5x -> 1.0x)
    print("\n5. Robustness Check under Degradation (Blur sigma=1.5 + Downscale/Upscale 0.5x)...")
    
    class PerturbedDataset(ArtVisionMultiTaskDataset):
        def __getitem__(self, idx: int):
            row = self.data.iloc[idx]
            img_path = str(row["filepath"])
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_dir, f"{row['image_id']}.jpg")
                if not os.path.exists(img_path):
                    img_path = os.path.join(self.image_dir, f"{row['image_id']}.png")
            
            with Image.open(img_path).convert("RGB") as pil_img:
                # Apply downscale-upscale
                w, h = pil_img.size
                small = pil_img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BILINEAR)
                upscaled = small.resize((w, h), Image.Resampling.BILINEAR)
                
                # Apply Gaussian blur
                arr = np.array(upscaled, dtype=float)
                blurred_arr = gaussian_filter(arr, sigma=(1.5, 1.5, 0))
                perturbed_img = Image.fromarray(np.clip(blurred_arr, 0, 255).astype(np.uint8))
                
            tensor = self.transform(perturbed_img)
            target = 1.0 if bool(row["is_ai_generated"]) else 0.0
            return tensor, target
            
    perturbed_ds = PerturbedDataset(
        metadata_path=metadata_path,
        image_dir="data/processed/images",
        split="test",
        transform=transform_eval
    )
    perturbed_loader = DataLoader(perturbed_ds, batch_size=32, shuffle=False)
    
    pert_preds = []
    pert_probs = []
    pert_targets = []
    
    with torch.no_grad():
        for imgs, tgts in perturbed_loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            logits = outs["authenticity"].squeeze(-1)
            probs = torch.sigmoid(logits / auth_T).cpu().numpy()
            pert_probs.extend(probs)
            pert_preds.extend((probs >= 0.5).astype(int))
            pert_targets.extend(tgts.numpy())
            
    pert_probs = np.array(pert_probs)
    pert_preds = np.array(pert_preds)
    pert_targets = np.array(pert_targets)
    
    pert_acc = accuracy_score(pert_targets, pert_preds)
    pert_f1 = f1_score(pert_targets, pert_preds, average="macro")
    pert_auroc = roc_auc_score(pert_targets, pert_probs)
    
    print(f"  Degraded Test Accuracy: {pert_acc*100:.2f}% (drop of {(0.9706 - pert_acc)*100:+.2f}%)")
    print(f"  Degraded Macro-F1:      {pert_f1*100:.2f}% (drop of {(0.9349 - pert_f1)*100:+.2f}%)")
    print(f"  Degraded AUROC:         {pert_auroc:.4f} (drop of {(0.9951 - pert_auroc):+.4f})")
    cm_pert = confusion_matrix(pert_targets, pert_preds)
    print("  Confusion Matrix under degradation [0=Human, 1=AI]:")
    print(f"    Human: TN={cm_pert[0,0]}, FP={cm_pert[0,1]}")
    print(f"    AI:    FN={cm_pert[1,0]}, TP={cm_pert[1,1]}")

    print("\n" + "="*60)
    print("STEP 0c: UNCERTAINTY THRESHOLD SWEEP (ON VAL SET)")
    print("="*60)
    
    val_ds = ArtVisionMultiTaskDataset(
        metadata_path=metadata_path,
        image_dir="data/processed/images",
        split="val",
        transform=transform_eval
    )
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
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
                
    val_logits = {t: torch.cat(val_logits[t]) for t in val_logits}
    val_targets = {t: torch.cat(val_targets[t]) for t in val_targets}
    val_masks = {t: torch.cat(val_masks[t]) for t in val_masks}
    
    threshold_sweeps = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for t in model.enabled_tasks:
        msk = val_masks[t].bool().numpy()
        if not msk.any():
            continue
        v_l = val_logits[t][msk]
        v_tgt = val_targets[t][msk].numpy()
        T = scaler.temperatures.get(t, 1.0)
        
        print(f"\nTask: {t.upper()} (Validation samples: {len(v_tgt)})")
        print(f"{'Threshold':<12} | {'Coverage':<12} | {'Selective Acc':<14} | {'Margin Thresh':<14}")
        print("-" * 58)
        
        if t == "authenticity":
            probs = torch.sigmoid(v_l / T).numpy()
            conf = np.maximum(probs, 1.0 - probs)
            preds = (probs >= 0.5).astype(int)
            for th in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                cov_mask = conf >= th
                cov = np.mean(cov_mask)
                sel_acc = accuracy_score(v_tgt[cov_mask], preds[cov_mask]) if cov > 0 else 0.0
                print(f"{th:<12.2f} | {cov*100:<11.1f}% | {sel_acc*100:<13.2f}% | margin >= {2*(th-0.5):.2f}")
        else:
            probs = F.softmax(v_l / T, dim=-1).numpy()
            sorted_p = np.sort(probs, axis=-1)
            conf = sorted_p[:, -1]
            margin = sorted_p[:, -1] - sorted_p[:, -2]
            preds = np.argmax(probs, axis=-1)
            for th in threshold_sweeps:
                cov_mask = conf >= th
                cov = np.mean(cov_mask)
                sel_acc = accuracy_score(v_tgt[cov_mask], preds[cov_mask]) if cov > 0 else 0.0
                print(f"{th:<12.2f} | {cov*100:<11.1f}% | {sel_acc*100:<13.2f}% | (baseline {accuracy_score(v_tgt, preds)*100:.1f}%)")

    print("\n" + "="*60)
    print("STEP 0d: MOVEMENT MAPPING AUDIT & PROPOSAL")
    print("="*60)
    with open("configs/label_maps.yaml") as f:
        maps = yaml.safe_load(f)
    mov_map = maps.get("movement_map", {})
    print("Current movement_map styles count:", len(mov_map))
    
    unmapped = [s for s in STYLE_CLASSES if s not in mov_map]
    print(f"Unmapped styles from STYLE_CLASSES ({len(unmapped)}):", unmapped)
    
    print("\n" + "="*60)
    print("STEP 0e: ORPHAN IMAGE INTEGRITY CHECK")
    print("="*60)
    # Check wikiart_006001.jpg to wikiart_006772.jpg
    orphan_ids = [f"wikiart_{i:06d}" for i in range(6001, 6773)]
    metadata_ids = set(df["image_id"].values)
    in_metadata = [oid for oid in orphan_ids if oid in metadata_ids]
    print(f"Total orphan image files checked: {len(orphan_ids)}")
    print(f"Orphan images present in metadata.csv: {len(in_metadata)}")
    if len(in_metadata) == 0:
        print("[PASS] Confirmed: None of wikiart_006001.jpg - wikiart_006772.jpg are in metadata.csv.")
    else:
        print("[FAIL] Warning: Found orphan IDs in metadata.csv!")

if __name__ == "__main__":
    run_step0()
