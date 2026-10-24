import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from src.data.dataset import StyleDataset
from src.models.style_model import StyleModel
from src.training.seed import seed_everything
from src.training.losses import get_criterion
from src.training.metrics import calculate_metrics
import time
import json

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": loss.item()})
        
    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(dataloader), metrics

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    pbar = tqdm(dataloader, desc="Validating")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(dataloader), metrics

def main(config_path, dev_run=False):
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    seed_everything(config["training"]["seed"])
    device = torch.device(config["training"]["device"] if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    train_ds = StyleDataset(config["data"]["metadata_path"], config["data"]["image_dir"], 
                            config["data"]["train_split"], transform_train, include_ai=False)
    val_ds = StyleDataset(config["data"]["metadata_path"], config["data"]["image_dir"], 
                          config["data"]["val_split"], transform_eval, include_ai=False)
                          
    if dev_run:
        # subset of 2 batches
        train_ds = torch.utils.data.Subset(train_ds, range(2 * config["data"]["batch_size"]))
        val_ds = torch.utils.data.Subset(val_ds, range(2 * config["data"]["batch_size"]))
        
    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=config["data"]["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=config["data"]["num_workers"])
    
    class_counts = train_ds.dataset.get_class_counts() if isinstance(train_ds, torch.utils.data.Subset) else train_ds.get_class_counts()
    criterion = get_criterion(class_counts, config["training"]["label_smoothing"]).to(device)
    
    model = StyleModel(config["model"]["backbone"], config["model"]["num_classes"], config["model"]["pretrained"])
    model.to(device)
    
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    history = []
    best_val_f1 = 0
    patience_counter = 0
    
    start_time = time.time()
    
    # Stage 1: Freeze backbone
    print("--- Stage 1: Training Head Only ---")
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config["training"]["lr_head"], weight_decay=config["training"]["weight_decay"])
    
    epochs_stage1 = 2 if dev_run else config["training"]["epochs_stage1"]
    for epoch in range(epochs_stage1):
        t_loss, t_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = validate(model, val_loader, criterion, device)
        
        train_acc = t_metrics["accuracy"]
        val_macro_f1 = v_metrics["macro_f1"]
        print(f"Epoch {epoch+1}/{epochs_stage1} - Train Loss: {t_loss:.4f} Acc: {train_acc:.4f} - Val Loss: {v_loss:.4f} Macro-F1: {val_macro_f1:.4f}")
        
        history.append({"stage": 1, "epoch": epoch+1, "train_loss": t_loss, "val_loss": v_loss, "val_macro_f1": v_metrics["macro_f1"]})
        
    # Stage 2: Unfreeze backbone
    print("--- Stage 2: Fine-tuning Full Model ---")
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config["training"]["lr_backbone"]},
        {"params": model.head.parameters(), "lr": config["training"]["lr_head"]}
    ], weight_decay=config["training"]["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs_stage2"])
    
    epochs_stage2 = 2 if dev_run else config["training"]["epochs_stage2"]
    for epoch in range(epochs_stage2):
        t_loss, t_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        train_acc = t_metrics["accuracy"]
        val_macro_f1 = v_metrics["macro_f1"]
        print(f"Epoch {epoch+1}/{epochs_stage2} - Train Loss: {t_loss:.4f} Acc: {train_acc:.4f} - Val Loss: {v_loss:.4f} Macro-F1: {val_macro_f1:.4f}")
        
        history.append({"stage": 2, "epoch": epoch+1, "train_loss": t_loss, "val_loss": v_loss, "val_macro_f1": v_metrics["macro_f1"]})
        
        if v_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = v_metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(config["training"]["output_dir"], "best_model.pth"))
            patience_counter = 0
            print("  [*] Saved best model!")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                print("Early stopping triggered!")
                break
                
    end_time = time.time()
    
    with open(os.path.join(config["training"]["output_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    if dev_run:
        est_time = (end_time - start_time) / 4 * (config["training"]["epochs_stage1"] + config["training"]["epochs_stage2"]) * (len(train_ds.dataset) / config["data"]["batch_size"] / 2)
        print(f"Dev run complete. Estimated full run time: {est_time/60:.2f} minutes.")
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_style.yaml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    main(args.config, args.dev)
