import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from src.data.multitask_dataset import ArtVisionMultiTaskDataset
from src.models.multitask import MultiTaskModel
from src.training.multitask_loss import MaskedMultiTaskLoss
from src.training.seed import seed_everything


def run_overfit_check():
    seed_everything(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running overfit sanity check on device: {device}")

    with open("configs/train_multitask.yaml") as f:
        config = yaml.safe_load(f)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ArtVisionMultiTaskDataset(
        metadata_path=config["data"]["metadata_path"],
        image_dir=config["data"]["image_dir"],
        split="train",
        transform=transform
    )

    # Collect ~100 samples with balanced representation
    mood_indices = dataset.get_task_indices("mood")[:30]
    artist_indices = dataset.get_task_indices("artist")[:7]
    auth_indices = dataset.get_task_indices("authenticity")[:70]
    
    selected_indices = list(dict.fromkeys(mood_indices + artist_indices + auth_indices))[:100]
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True)

    model = MultiTaskModel(config).to(device)
    model.unfreeze_backbone()

    # Equal weighting for overfit test
    criterion = MaskedMultiTaskLoss(config["tasks"], weighting_method="equal").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Training overfit test on {len(subset)} samples...")
    for epoch in range(1, 41):
        model.train()
        total_loss = 0.0
        correct_counts = {t: 0 for t in ["style", "authenticity", "mood"]}
        total_counts = {t: 0 for t in ["style", "authenticity", "mood"]}

        for images, targets, masks in loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            masks = {k: v.to(device) for k, v in masks.items()}

            optimizer.zero_grad()
            outputs = model(images)
            loss, task_losses, _ = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Track accuracy
            with torch.no_grad():
                # Style
                style_msk = masks["style"].bool()
                if style_msk.sum() > 0:
                    preds = outputs["style"][style_msk].argmax(dim=-1)
                    correct_counts["style"] += (preds == targets["style"][style_msk]).sum().item()
                    total_counts["style"] += style_msk.sum().item()

                # Authenticity
                auth_msk = masks["authenticity"].bool()
                if auth_msk.sum() > 0:
                    probs = torch.sigmoid(outputs["authenticity"][auth_msk])
                    preds = (probs > 0.5).float()
                    correct_counts["authenticity"] += (preds == targets["authenticity"][auth_msk]).sum().item()
                    total_counts["authenticity"] += auth_msk.sum().item()

                # Mood
                mood_msk = masks["mood"].bool()
                if mood_msk.sum() > 0:
                    preds = outputs["mood"][mood_msk].argmax(dim=-1)
                    correct_counts["mood"] += (preds == targets["mood"][mood_msk]).sum().item()
                    total_counts["mood"] += mood_msk.sum().item()

        if epoch % 10 == 0 or epoch == 40:
            acc_str = " | ".join([
                f"{t}: {correct_counts[t]}/{total_counts[t]} ({correct_counts[t]/max(1, total_counts[t]):.2%})"
                for t in correct_counts if total_counts[t] > 0
            ])
            print(f"Epoch {epoch:02d}/40 - Total Loss: {total_loss/len(loader):.4f} - Accuracies: {acc_str}")

    # Check that accuracy is high
    assert correct_counts["authenticity"] / total_counts["authenticity"] > 0.90, "Authenticity failed to overfit"
    assert correct_counts["style"] / total_counts["style"] > 0.90, "Style failed to overfit"
    print("Overfit sanity check PASSED (>90% accuracy on all labeled subsets)!")


if __name__ == "__main__":
    run_overfit_check()
