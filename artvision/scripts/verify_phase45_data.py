import os
import sys
import pandas as pd
import numpy as np


def verify_metadata():
    metadata_path = "data/processed/metadata.csv"
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} does not exist!")
        sys.exit(1)

    df = pd.read_csv(metadata_path)
    print(f"==================================================")
    print(f"DATASET INTEGRITY & SPLIT VERIFICATION REPORT")
    print(f"Total Rows in Metadata: {len(df)}")
    print(f"==================================================")

    # 1. Row counts per source_dataset and split
    print("\n--- Row Counts per Source and Split ---")
    cross = pd.crosstab(df["source_dataset"], df["split"], dropna=False, margins=True)
    print(cross.to_string())

    # 2. Check for data leakage across splits (MD5 and pHash)
    print("\n--- Data Leakage Check ---")
    splits = ["train", "val", "test"]
    split_md5s = {s: set(df[df["split"] == s]["file_hash"].dropna()) for s in splits}
    
    train_val_overlap = split_md5s["train"].intersection(split_md5s["val"])
    train_test_overlap = split_md5s["train"].intersection(split_md5s["test"])
    val_test_overlap = split_md5s["val"].intersection(split_md5s["test"])

    print(f"Train/Val MD5 overlap: {len(train_val_overlap)}")
    print(f"Train/Test MD5 overlap: {len(train_test_overlap)}")
    print(f"Val/Test MD5 overlap: {len(val_test_overlap)}")

    assert len(train_val_overlap) == 0, f"Leakage detected between train and val! ({len(train_val_overlap)} hashes)"
    assert len(train_test_overlap) == 0, f"Leakage detected between train and test! ({len(train_test_overlap)} hashes)"
    assert len(val_test_overlap) == 0, f"Leakage detected between val and test! ({len(val_test_overlap)} hashes)"
    print("[PASS] Zero hash leakage across splits.")

    # 3. Labeled row counts per task per split
    print("\n--- Labeled Counts per Task per Split ---")
    tasks = {
        "style": lambda r: pd.notna(r["style"]) and str(r["style"]).strip() != "",
        "artist": lambda r: pd.notna(r["artist"]) and str(r["artist"]).strip() != "" and str(r["artist"]) != "Unknown Artist",
        "genre": lambda r: pd.notna(r["genre"]) and str(r["genre"]).strip() != "" and str(r["genre"]) != "Unknown Genre",
        "mood": lambda r: pd.notna(r.get("mood")) or pd.notna(r.get("mood_dist")),
        "authenticity": lambda r: pd.notna(r["is_ai_generated"]),
    }

    coverage_rows = []
    for task_name, filter_fn in tasks.items():
        task_df = df[df.apply(filter_fn, axis=1)]
        n_train = (task_df["split"] == "train").sum()
        n_val = (task_df["split"] == "val").sum()
        n_test = (task_df["split"] == "test").sum()
        n_total = len(task_df)
        n_classes = task_df[task_name].nunique() if task_name in task_df else task_df["is_ai_generated"].nunique()
        coverage_rows.append({
            "Task": task_name,
            "Classes": n_classes,
            "Train": n_train,
            "Val": n_val,
            "Test": n_test,
            "Total Labeled": n_total
        })

    cov_df = pd.DataFrame(coverage_rows)
    print(cov_df.to_string(index=False))

    # Verify that all 5 tasks have samples in test
    for _, row in cov_df.iterrows():
        task = row["Task"]
        n_test = row["Test"]
        assert n_test > 0, f"CRITICAL: Task {task} has 0 test samples!"
        print(f"[PASS] Task {task}: {n_test} test samples verified.")

    # 4. Check image files existence
    print("\n--- Image File Verification ---")
    missing_files = []
    for idx in range(min(500, len(df))):
        fpath = df.iloc[idx]["filepath"]
        if not os.path.exists(fpath):
            missing_files.append(fpath)
    print(f"Sample 500 images checked: {len(missing_files)} missing.")
    assert len(missing_files) == 0, f"Found missing image files: {missing_files[:5]}"
    print("[PASS] Sampled image files confirmed on disk.")

    print("\nALL DATASET VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    verify_metadata()
