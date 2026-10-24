import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from torch.utils.data import DataLoader
from src.data.multitask_dataset import ArtVisionMultiTaskDataset
from src.training.sampler import BalancedMultiTaskBatchSampler


def test_balanced_sampler_guarantees():
    dataset = ArtVisionMultiTaskDataset(
        metadata_path="data/processed/metadata.csv",
        image_dir="data/processed/images",
        split="train"
    )

    batch_size = 32
    min_samples = {
        "style": 4,
        "authenticity": 4,
        "mood": 2,
        "artist": 2,
        "genre": 2,  # genre has 0 samples, should be handled gracefully
    }

    sampler = BalancedMultiTaskBatchSampler(
        dataset,
        batch_size=batch_size,
        min_samples_per_task=min_samples,
        seed=42
    )

    assert len(sampler) > 0, "Sampler length must be > 0"

    # Test the first 5 batches
    for i, batch_indices in enumerate(sampler):
        if i >= 5:
            break
        assert len(batch_indices) == batch_size, f"Batch size must be {batch_size}, got {len(batch_indices)}"
        assert len(set(batch_indices)) == batch_size, "Batch indices must be unique within batch"

        # Check task coverage
        mood_count = 0
        artist_count = 0
        style_count = 0
        auth_count = 0

        for idx in batch_indices:
            _, _, masks = dataset[idx]
            if masks["mood"].item() == 1.0:
                mood_count += 1
            if masks["artist"].item() == 1.0:
                artist_count += 1
            if masks["style"].item() == 1.0:
                style_count += 1
            if masks["authenticity"].item() == 1.0:
                auth_count += 1

        assert mood_count >= min_samples["mood"], f"Batch {i} mood count {mood_count} < {min_samples['mood']}"
        assert artist_count >= min_samples["artist"], f"Batch {i} artist count {artist_count} < {min_samples['artist']}"
        assert style_count >= min_samples["style"], f"Batch {i} style count {style_count} < {min_samples['style']}"
        assert auth_count >= min_samples["authenticity"], f"Batch {i} auth count {auth_count} < {min_samples['authenticity']}"

    print("Balanced sampler guarantees passed!")


if __name__ == "__main__":
    test_balanced_sampler_guarantees()
