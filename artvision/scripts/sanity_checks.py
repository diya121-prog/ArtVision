import sys
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import StyleDataset
from src.models.style_model import StyleModel
from src.training.seed import seed_everything

def check_leakage(metadata_path):
    print("--- 1. Data Leakage Check ---")
    df = pd.read_csv(metadata_path)
    train_ids = set(df[df["split"] == "train"]["image_id"])
    val_ids = set(df[df["split"] == "val"]["image_id"])
    test_ids = set(df[df["split"] == "test"]["image_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Leakage between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Leakage between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Leakage between val and test!"
    print("No leakage found between splits (image_ids are mutually exclusive).")

def check_visual_batch(dataset, output_path):
    print(f"--- 2. Visual Batch Check (saving to {output_path}) ---")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(loader))
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    # Unnormalize
    inv_trans = transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
    ])
    
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = inv_trans(images[i])
            img = img.permute(1, 2, 0).clamp(0, 1).numpy()
            ax.imshow(img)
            ax.set_title(idx_to_class[labels[i].item()])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    print("Visual batch grid saved.")

def overfit_test(dataset, device):
    print("--- 3. Overfit Test on ~100 samples ---")
    subset = Subset(dataset, range(100))
    # Override transform to eval transform for overfit test to remove augmentations
    subset.dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    loader = DataLoader(subset, batch_size=32, shuffle=True)
    
    model = StyleModel(pretrained=True).to(device)
    model.freeze_backbone()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2) # higher LR for fast overfit
    
    model.train()
    for epoch in range(50):
        total_loss, correct, total = 0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        if epoch % 10 == 0 or epoch == 49:
            print(f"Epoch {epoch+1}/50 - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.4f}")
    if acc < 0.95:
        print("WARNING: Model failed to overfit on 100 samples!")
    else:
        print("Overfit test passed (Acc > 95%).")

    # Restore transform
    subset.dataset.transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def logistic_regression_baseline(train_ds, val_ds, device):
    print("--- 4. Majority Class & Logistic Regression Baseline ---")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # Majority class
    train_labels = [y for _, y in train_loader for y in y]
    val_labels = [y for _, y in val_loader for y in y]
    
    from collections import Counter
    majority_class = Counter(train_labels).most_common(1)[0][0]
    maj_acc = sum(1 for y in val_labels if y == majority_class) / len(val_labels)
    print(f"Majority Class Baseline Acc (Val): {maj_acc:.4f}")
    
    # Extract embeddings
    model = StyleModel(pretrained=True).to(device)
    model.eval()
    
    def extract_feats(loader):
        feats, lbls = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Extracting embeddings"):
                _, embeddings = model(images.to(device), return_embeddings=True)
                feats.append(embeddings.cpu())
                lbls.append(labels)
        return torch.cat(feats).numpy(), torch.cat(lbls).numpy()
    
    X_train, y_train = extract_feats(train_loader)
    X_val, y_val = extract_feats(val_loader)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"Logistic Regression on Frozen Embeddings Acc (Val): {accuracy_score(y_val, y_pred):.4f}")

if __name__ == "__main__":
    seed_everything(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    with open("configs/train_style.yaml") as f:
        config = yaml.safe_load(f)
        
    check_leakage(config["data"]["metadata_path"])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = StyleDataset(
        metadata_path=config["data"]["metadata_path"],
        image_dir=config["data"]["image_dir"],
        split=config["data"]["train_split"],
        transform=transform_train,
        include_ai=False
    )
    
    val_dataset = StyleDataset(
        metadata_path=config["data"]["metadata_path"],
        image_dir=config["data"]["image_dir"],
        split=config["data"]["val_split"],
        transform=transform_eval,
        include_ai=False
    )
    
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    check_visual_batch(train_dataset, os.path.join(config["training"]["output_dir"], "sanity_visual_batch.png"))
    overfit_test(train_dataset, device)
    logistic_regression_baseline(train_dataset, val_dataset, device)
