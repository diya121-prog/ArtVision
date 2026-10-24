import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.models.multitask import MultiTaskModel


class ArtworkEmbeddingDataset(Dataset):
    """
    Dataset strictly iterating over metadata.csv rows for embedding extraction.
    Never uses os.listdir to avoid indexing orphan images.
    """
    def __init__(self, metadata_path: str, image_dir: str, transform=None):
        self.df = pd.read_csv(metadata_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["image_id"])
        img_path = str(row.get("filepath", ""))
        
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.image_dir, f"{img_id}.png")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, img_id, idx


def build_embeddings(
    checkpoint_path: str = "runs/multitask_model/best_model.pth",
    metadata_path: str = "data/processed/metadata.csv",
    image_dir: str = "data/processed/images",
    output_dir: str = "runs/retrieval",
    batch_size: int = 64
):
    print(f"=== Extracting L2-Normalized Artwork Embeddings ===")
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]
    model = MultiTaskModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Transform
    img_size = config.get("data", {}).get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ArtworkEmbeddingDataset(metadata_path, image_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    metadata_records = []

    # Pre-extract metadata dicts aligned with rows
    df = dataset.df
    for i in range(len(df)):
        r = df.iloc[i]
        metadata_records.append({
            "image_id": str(r["image_id"]),
            "style": str(r["style"]) if pd.notna(r.get("style")) else "Unknown",
            "artist": str(r["artist"]) if pd.notna(r.get("artist")) else "Unknown",
            "genre": str(r["genre"]) if pd.notna(r.get("genre")) else "Unknown",
            "is_ai_generated": bool(r.get("is_ai_generated", False)),
            "filepath": str(r.get("filepath", f"data/processed/images/{r['image_id']}.jpg"))
        })

    with torch.no_grad():
        for images, img_ids, indices in tqdm(loader, desc="Extracting Embeddings"):
            images = images.to(device)
            # EfficientNet pooled 1280-d features
            feats = model.backbone(images) # [B, 1280]
            # L2-normalize
            feats_norm = F.normalize(feats, p=2, dim=-1)
            all_embeddings.append(feats_norm.cpu().numpy().astype(np.float32))

    embeddings_matrix = np.vstack(all_embeddings)
    print(f"Generated embeddings matrix: shape {embeddings_matrix.shape}, dtype {embeddings_matrix.dtype}")

    # Save to disk
    emb_path = os.path.join(output_dir, "embeddings.npy")
    ids_path = os.path.join(output_dir, "image_ids.json")

    np.save(emb_path, embeddings_matrix)
    with open(ids_path, "w") as f:
        json.dump(metadata_records, f, indent=2)

    print(f"Saved {len(embeddings_matrix)} embeddings to: {emb_path}")
    print(f"Saved {len(metadata_records)} metadata records to: {ids_path}")
    return emb_path, ids_path


if __name__ == "__main__":
    build_embeddings()
