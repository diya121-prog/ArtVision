import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import json
from src.data.multitask_dataset import ArtVisionMultiTaskDataset

def test_leakage():
    df = pd.read_csv("data/processed/metadata.csv")
    train_ph = set(df[df['split']=='train']['phash'].dropna())
    val_ph = set(df[df['split']=='val']['phash'].dropna())
    test_ph = set(df[df['split']=='test']['phash'].dropna())
    
    assert train_ph.isdisjoint(val_ph), "Train and Val overlap!"
    assert train_ph.isdisjoint(test_ph), "Train and Test overlap!"
    assert val_ph.isdisjoint(test_ph), "Val and Test overlap!"

def test_mask_and_soft_labels():
    dataset = ArtVisionMultiTaskDataset("data/processed/metadata.csv", image_dir="data/processed/images", split="train", transform=None)
    df = dataset.data
    
    # Check one ai_artbench sample if in train
    ai_rows = df[df['source_dataset'] == 'ai_artbench']
    if len(ai_rows) > 0:
        ai_idx = ai_rows.index[0]
        img, targets, masks = dataset[ai_idx]
        assert masks['mood'] == 0.0, "Mask should be 0 for missing mood"
        assert float(targets['mood_dist'].sum()) == 0.0, "Soft labels should be 0 for missing mood"
    
    # Check one wikiart_emotions sample if in train
    wiki_rows = df[df['source_dataset'] == 'wikiart_emotions']
    if len(wiki_rows) > 0:
        wiki_idx = wiki_rows.index[0]
        img, targets, masks = dataset[wiki_idx]
        assert masks['mood'] == 1.0, "Mask should be 1 for valid mood"
        soft_sum = float(targets['mood_dist'].sum())
        assert abs(soft_sum - 1.0) < 1e-4 or abs(soft_sum) < 1e-4, f"Soft labels sum to {soft_sum}"

if __name__ == "__main__":
    test_leakage()
    print("test_leakage passed!")
    test_mask_and_soft_labels()
    print("test_mask_and_soft_labels passed!")


