import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
from src.training.train_multitask import train_multitask
from src.training.evaluate_multitask import evaluate_multitask_model


def run_all():
    config_path = "configs/train_multitask.yaml"

    results = {
        "single_task": {},
        "incremental": {},
        "loss_weighting_ablation": {},
    }

    # ================= 1. Single-task baselines =================
    print("\n>>> Running Single-Task Baseline: Authenticity")
    ckpt_auth, hist_auth = train_multitask(config_path, enabled_tasks_list=["authenticity"], output_dir="runs/single_task_auth")
    eval_auth = evaluate_multitask_model(ckpt_auth, output_dir="runs/single_task_auth/eval", split="val")
    results["single_task"]["authenticity"] = eval_auth.get("authenticity", {})

    print("\n>>> Running Single-Task Baseline: Mood")
    ckpt_mood, hist_mood = train_multitask(config_path, enabled_tasks_list=["mood"], output_dir="runs/single_task_mood")
    eval_mood = evaluate_multitask_model(ckpt_mood, output_dir="runs/single_task_mood/eval", split="val")
    results["single_task"]["mood"] = eval_mood.get("mood", {})

    print("\n>>> Running Single-Task Baseline: Artist")
    ckpt_artist, hist_artist = train_multitask(config_path, enabled_tasks_list=["artist"], output_dir="runs/single_task_artist")
    eval_artist = evaluate_multitask_model(ckpt_artist, output_dir="runs/single_task_artist/eval", split="val")
    results["single_task"]["artist"] = eval_artist.get("artist", {})

    # Single-task style from Phase 3: Val F1 = 0.2558, Test Acc = 0.4900, Test F1 = 0.4722
    results["single_task"]["style"] = {
        "accuracy": 0.4900,
        "macro_f1": 0.4722,
        "source": "Phase 3 Best Checkpoint"
    }

    # ================= 2. Incremental Steps =================
    print("\n>>> Incremental Step 1: style + genre")
    ckpt_step1, _ = train_multitask(config_path, enabled_tasks_list=["style", "genre"], weighting_method="equal", output_dir="runs/incremental_1_style_genre")
    eval_step1 = evaluate_multitask_model(ckpt_step1, output_dir="runs/incremental_1_style_genre/eval", split="val")
    results["incremental"]["step1_style_genre"] = eval_step1

    print("\n>>> Incremental Step 2: + artist")
    ckpt_step2, _ = train_multitask(config_path, enabled_tasks_list=["style", "genre", "artist"], weighting_method="equal", output_dir="runs/incremental_2_artist")
    eval_step2 = evaluate_multitask_model(ckpt_step2, output_dir="runs/incremental_2_artist/eval", split="val")
    results["incremental"]["step2_plus_artist"] = eval_step2

    print("\n>>> Incremental Step 3: + mood")
    ckpt_step3, _ = train_multitask(config_path, enabled_tasks_list=["style", "genre", "artist", "mood"], weighting_method="equal", output_dir="runs/incremental_3_mood")
    eval_step3 = evaluate_multitask_model(ckpt_step3, output_dir="runs/incremental_3_mood/eval", split="val")
    results["incremental"]["step3_plus_mood"] = eval_step3

    print("\n>>> Incremental Step 4 / Ablation: Equal Weighting (Full Model)")
    ckpt_step4, _ = train_multitask(config_path, enabled_tasks_list=["style", "genre", "artist", "mood", "authenticity"], weighting_method="equal", output_dir="runs/ablation_equal")
    eval_step4 = evaluate_multitask_model(ckpt_step4, output_dir="runs/ablation_equal/eval", split="val")
    results["incremental"]["step4_plus_authenticity"] = eval_step4
    results["loss_weighting_ablation"]["equal"] = eval_step4

    # ================= 3. Loss Weighting Ablation =================
    print("\n>>> Loss Weighting Ablation: Manual Weighting")
    # Manual weights: emphasize style and authenticity, lower weight on small mood/artist heads
    ckpt_manual, _ = train_multitask(config_path, enabled_tasks_list=["style", "genre", "artist", "mood", "authenticity"], weighting_method="manual", output_dir="runs/ablation_manual")
    eval_manual = evaluate_multitask_model(ckpt_manual, output_dir="runs/ablation_manual/eval", split="val")
    results["loss_weighting_ablation"]["manual"] = eval_manual

    print("\n>>> Loss Weighting Ablation: Uncertainty Weighting (Kendall et al.)")
    ckpt_uncert, _ = train_multitask(config_path, enabled_tasks_list=["style", "genre", "artist", "mood", "authenticity"], weighting_method="uncertainty", output_dir="runs/multitask_model")
    eval_uncert_val = evaluate_multitask_model(ckpt_uncert, output_dir="runs/multitask_model/eval_val", split="val")
    results["loss_weighting_ablation"]["uncertainty"] = eval_uncert_val

    # ================= 4. Final Evaluation on TEST set =================
    print("\n>>> FINAL EVALUATION ON TEST SET (Full Multi-Task Model)")
    test_results = evaluate_multitask_model(ckpt_uncert, output_dir="runs/multitask_model/evaluation", split="test")
    results["final_test_evaluation"] = test_results

    # Save all summary results
    with open("runs/experiment_summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n=======================================================")
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("Results saved to runs/experiment_summary.json")
    print("=======================================================\n")


if __name__ == "__main__":
    run_all()
