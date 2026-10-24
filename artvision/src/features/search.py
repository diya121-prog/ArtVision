import os
import json
from typing import List, Dict, Any, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.models.multitask import MultiTaskModel


class ArtworkSearcher:
    """
    Lightweight cosine-similarity artwork retrieval system.
    Loads L2-normalized embeddings and metadata.
    Enforces self-exclusion and near-duplicate filtering (similarity > 0.995).
    """
    def __init__(
        self,
        embeddings_path: str = "runs/retrieval/embeddings.npy",
        image_ids_path: str = "runs/retrieval/image_ids.json",
        checkpoint_path: str = "runs/multitask_model/best_model.pth",
        device: str = None
    ):
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found at: {embeddings_path}")
        if not os.path.exists(image_ids_path):
            raise FileNotFoundError(f"Metadata not found at: {image_ids_path}")

        self.embeddings = np.load(embeddings_path) # [N, 1280]
        with open(image_ids_path, "r") as f:
            self.metadata = json.load(f)

        self.id_to_idx = {r["image_id"]: i for i, r in enumerate(self.metadata)}
        self.device = torch.device(device if device else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _get_model(self):
        if self._model is None:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            config = ckpt["config"]
            model = MultiTaskModel(config)
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(self.device)
            model.eval()
            self._model = model
        return self._model

    def extract_query_embedding(self, query: Union[str, Image.Image, torch.Tensor]) -> np.ndarray:
        """
        Extracts L2-normalized 1280-d embedding for a query image (path, PIL, or tensor).
        """
        if isinstance(query, str):
            image = Image.open(query).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self.device)
        elif isinstance(query, Image.Image):
            tensor = self._transform(query.convert("RGB")).unsqueeze(0).to(self.device)
        elif isinstance(query, torch.Tensor):
            if query.dim() == 3:
                tensor = query.unsqueeze(0).to(self.device)
            else:
                tensor = query.to(self.device)
        else:
            raise TypeError(f"Unsupported query type: {type(query)}")

        model = self._get_model()
        with torch.no_grad():
            emb = model.backbone(tensor)
            emb_norm = F.normalize(emb, p=2, dim=-1)
        return emb_norm.squeeze(0).cpu().numpy().astype(np.float32)

    def find_similar(
        self,
        query: Union[str, Image.Image, torch.Tensor],
        top_k: int = 5,
        min_sim: float = 0.0,
        max_sim: float = 0.995
    ) -> List[Dict[str, Any]]:
        """
        Finds top_k similar artworks using cosine similarity.
        - If query is an existing image_id, uses cached embedding and excludes self.
        - Excludes near-duplicates with cosine similarity > max_sim (0.995).
        """
        query_id = None
        if isinstance(query, str) and query in self.id_to_idx:
            # Query is an existing image_id in catalog
            query_id = query
            idx = self.id_to_idx[query_id]
            q_emb = self.embeddings[idx]
        else:
            # Query is an image path or PIL image
            if isinstance(query, str) and os.path.exists(query):
                # Check if this filepath matches an existing image
                q_emb = self.extract_query_embedding(query)
            elif isinstance(query, (Image.Image, torch.Tensor)):
                q_emb = self.extract_query_embedding(query)
            else:
                raise ValueError(f"Query '{query}' is neither an image path nor a recognized catalog image_id.")

        # Cosine similarity (since both embeddings and q_emb are unit-normalized)
        sims = np.dot(self.embeddings, q_emb)

        # Build candidate indices
        sorted_indices = np.argsort(sims)[::-1]

        results = []
        for idx in sorted_indices:
            cand = self.metadata[idx]
            sim = float(sims[idx])

            # Self-exclusion
            if query_id is not None and cand["image_id"] == query_id:
                continue

            # Near-duplicate filtering (cosine > 0.995)
            if sim > max_sim:
                continue

            # Minimum similarity threshold
            if sim < min_sim:
                break

            results.append({
                "image_id": cand["image_id"],
                "similarity": round(sim, 4),
                "style": cand.get("style", "Unknown"),
                "artist": cand.get("artist", "Unknown"),
                "genre": cand.get("genre", "Unknown"),
                "is_ai_generated": cand.get("is_ai_generated", False),
                "filepath": cand.get("filepath", "")
            })

            if len(results) >= top_k:
                break

        return results
