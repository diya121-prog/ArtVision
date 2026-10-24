import os
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.data.dataset import StyleDataset
from src.models.style_model import StyleModel
from src.training.metrics import calculate_metrics
import numpy as np
import pandas as pd

@torch.no_grad()
def evaluate_model(config_path, checkpoint_path, output_dir, split="test", include_ai=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.backends.mps.is_available() else "cpu")
    
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config["data"]["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = StyleDataset(config["data"]["metadata_path"], config["data"]["image_dir"], 
                           split, transform_eval, include_ai=include_ai)
                           
    dataloader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
    model = StyleModel(config["model"]["backbone"], config["model"]["num_classes"], pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc=f"Evaluating {split} (AI={include_ai})")
    for images, labels in pbar:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = calculate_metrics(all_labels, all_preds, all_probs)
    
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_ai" if include_ai else "_human"
    
    # 1. Classification report
    classes = dataset.classes
    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, f"classification_report_{split}{suffix}.csv"))
    
    # 2. Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix - {split.capitalize()} {suffix.strip('_')}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{split}{suffix}.png"))
    plt.close()
    
    # 3. Top confused pairs
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((classes[i], classes[j], cm[i, j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    with open(os.path.join(output_dir, f"top_confused_pairs_{split}{suffix}.txt"), "w") as f:
        f.write("True -> Predicted : Count\\n")
        for true, pred, count in confused_pairs[:10]:
            f.write(f"{true} -> {pred} : {count}\\n")
            
    # 4. Calibration
    correct_mask = all_preds == all_labels
    confidences = np.max(all_probs, axis=1)
    avg_conf_correct = np.mean(confidences[correct_mask]) if sum(correct_mask) > 0 else 0
    avg_conf_wrong = np.mean(confidences[~correct_mask]) if sum(~correct_mask) > 0 else 0
    
    metrics["avg_conf_correct"] = avg_conf_correct
    metrics["avg_conf_wrong"] = avg_conf_wrong
    
    print(f"\\n--- Metrics for {split} {suffix} ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # 5. Misclassifications Grid
    wrong_mask = all_preds != all_labels
    wrong_idx = np.where(wrong_mask)[0]
    
    if len(wrong_idx) > 0:
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        # Unnormalize
        inv_trans = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
        ])
        
        for i, ax in enumerate(axes.flat):
            if i < len(wrong_idx) and i < 16:
                idx = wrong_idx[i]
                img, _ = dataset[idx]
                img = inv_trans(img)
                img = img.permute(1, 2, 0).clamp(0, 1).numpy()
                ax.imshow(img)
                
                true_lbl = classes[all_labels[idx]]
                pred_lbl = classes[all_preds[idx]]
                conf = confidences[idx] * 100
                
                ax.set_title(f"T: {true_lbl}\\nP: {pred_lbl}\\nConf: {conf:.1f}%", fontsize=9, color="red")
            ax.axis("off")
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"misclassifications_{split}{suffix}.png"))
        plt.close()

    return metrics, all_preds, all_labels, all_probs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_style.yaml")
    parser.add_argument("--checkpoint", type=str, default="runs/style_model/best_model.pth")
    parser.add_argument("--output_dir", type=str, default="runs/style_model/evaluation")
    args = parser.parse_args()
    
    # Evaluate on Human test set
    evaluate_model(args.config, args.checkpoint, args.output_dir, split="test", include_ai=False)
    
    # Evaluate on AI test set
    evaluate_model(args.config, args.checkpoint, args.output_dir, split="test", include_ai=True)
