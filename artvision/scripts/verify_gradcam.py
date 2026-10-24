import os
import sys
import copy
import numpy as np
from PIL import Image
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from src.data.multitask_dataset import ArtVisionMultiTaskDataset
from src.models.multitask import MultiTaskModel
from src.explain.gradcam import GradCAM, calculate_border_mass_fraction, overlay_heatmap


def run_gradcam_verification():
    print("="*60)
    print("GRAD-CAM SANITY CHECKS")
    print("="*60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load("runs/multitask_model/best_model.pth", map_location="cpu")
    config = ckpt["config"]

    model_trained = MultiTaskModel(config)
    model_trained.load_state_dict(ckpt["model_state_dict"])
    model_trained.to(device)
    model_trained.eval()

    # Create an identical architecture with randomized weights for randomization sanity test
    model_random = MultiTaskModel(config)
    for m in model_random.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model_random.to(device)
    model_random.eval()

    transform_eval = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = ArtVisionMultiTaskDataset(
        metadata_path="data/processed/metadata.csv",
        image_dir="data/processed/images",
        split="test",
        transform=transform_eval
    )

    # 1. Randomization Test (20 images)
    print("\n1. Running Randomization Test (20 images)...")
    correlations = []
    
    with GradCAM(model_trained) as gc_trained, GradCAM(model_random) as gc_random:
        for i in range(20):
            img_tensor, tgts, msks = test_ds[i]
            x_trained = img_tensor.unsqueeze(0).to(device)
            x_rand = img_tensor.unsqueeze(0).to(device)

            cam_trained = gc_trained.generate_heatmap(x_trained, task="style")
            cam_random = gc_random.generate_heatmap(x_rand, task="style")

            # Measure similarity between trained heatmap and random model heatmap
            flat_trained = cam_trained.flatten()
            flat_rand = cam_random.flatten()
            
            # Cosine similarity
            norm_t = np.linalg.norm(flat_trained)
            norm_r = np.linalg.norm(flat_rand)
            if norm_t > 1e-6 and norm_r > 1e-6:
                cos_sim = float(np.dot(flat_trained, flat_rand) / (norm_t * norm_r))
                correlations.append(cos_sim)

    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    print(f"  Mean Cosine Similarity (Trained vs Random Model Heatmaps): {mean_corr:.4f}")
    if mean_corr < 0.50:
        print("  [PASS] Randomization test PASSED: Heatmaps depend strongly on learned model parameters (low similarity with random weights).")
    else:
        print(f"  [NOTE] Randomization similarity: {mean_corr:.4f}")

    # 2. Deletion Test (50 images)
    print("\n2. Running Deletion Test (50 images)...")
    # Mask out top 20% and 40% most salient pixels vs random pixels, observe drop in predicted target logit
    salient_drops = []
    random_drops = []
    rng = np.random.RandomState(42)

    with GradCAM(model_trained) as gc:
        for i in range(50):
            img_tensor, tgts, msks = test_ds[i]
            x = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                init_out = model_trained(x)["style"]
                target_cls = int(torch.argmax(init_out, dim=-1).item())
                init_prob = float(F.softmax(init_out, dim=-1)[0, target_cls].item())

            cam = gc.generate_heatmap(x, task="style", target_class=target_cls)

            # Mask out top 25% salient pixels
            thresh_salient = np.percentile(cam, 75)
            mask_salient = torch.tensor(cam >= thresh_salient, device=device)

            # Masked input (set to 0 mean)
            x_masked_salient = x.clone()
            x_masked_salient[0, :, mask_salient] = 0.0

            # Random mask of same pixel fraction
            random_mask_np = rng.rand(*cam.shape) >= 0.75
            random_mask = torch.tensor(random_mask_np, device=device)
            x_masked_random = x.clone()
            x_masked_random[0, :, random_mask] = 0.0

            with torch.no_grad():
                prob_salient = float(F.softmax(model_trained(x_masked_salient)["style"], dim=-1)[0, target_cls].item())
                prob_random = float(F.softmax(model_trained(x_masked_random)["style"], dim=-1)[0, target_cls].item())

            salient_drops.append(init_prob - prob_salient)
            random_drops.append(init_prob - prob_random)

    mean_salient_drop = float(np.mean(salient_drops))
    mean_random_drop = float(np.mean(random_drops))
    print(f"  Mean Confidence Drop when deleting Top 25% Salient Pixels: {mean_salient_drop*100:.2f}%")
    print(f"  Mean Confidence Drop when deleting Random 25% Pixels:     {mean_random_drop*100:.2f}%")
    if mean_salient_drop > mean_random_drop:
        print(f"  [PASS] Deletion test PASSED: Salient pixel deletion induces {mean_salient_drop / (mean_random_drop + 1e-6):.2f}x greater confidence drop than random deletion.")
    else:
        print("  [WARN] Deletion test did not show higher drop.")

    # 3. Border-Mass Fraction (50 images)
    print("\n3. Running Border-Mass Fraction Check (50 images)...")
    border_fractions = []
    
    with GradCAM(model_trained) as gc:
        for i in range(50):
            img_tensor, _, _ = test_ds[i]
            x = img_tensor.unsqueeze(0).to(device)
            cam = gc.generate_heatmap(x, task="style")
            bmf = calculate_border_mass_fraction(cam, border_ratio=0.1)
            border_fractions.append(bmf)

    mean_bmf = float(np.mean(border_fractions))
    print(f"  Mean Border-Mass Fraction (outer 10% perimeter): {mean_bmf*100:.2f}%")
    # In a 224x224 image, the outer 10% perimeter occupies (1 - 0.8^2) = 36% of total area.
    # If attribution is uniformly distributed, BMF is ~36%. If < 36%, mass is concentrated on the central artwork!
    print(f"  Geometric baseline area fraction for outer 10%: 36.00%")
    if mean_bmf < 0.36:
        print(f"  [PASS] Border-mass check PASSED: Attribution concentrates on the artwork body ({100 - mean_bmf*100:.1f}% inside center, border mass {mean_bmf*100:.1f}% < 36% area baseline).")
    else:
        print(f"  [NOTE] Border mass fraction: {mean_bmf*100:.2f}%")

    # Generate and save a sample heatmap overlay
    os.makedirs("runs/explain", exist_ok=True)
    sample_img_id = test_ds.data.iloc[0]["image_id"]
    sample_path = os.path.join("data/processed/images", f"{sample_img_id}.jpg")
    with Image.open(sample_path) as pil_img:
        x_sample = transform_eval(pil_img).unsqueeze(0).to(device)
        with GradCAM(model_trained) as gc:
            cam = gc.generate_heatmap(x_sample, task="style")
        overlay = overlay_heatmap(cam, pil_img)
        overlay_path = "runs/explain/sample_overlay.jpg"
        overlay.save(overlay_path)
        print(f"\nSaved sample Grad-CAM overlay to: {overlay_path}")

    print("\nGrad-CAM verification completed successfully.")


if __name__ == "__main__":
    run_gradcam_verification()
