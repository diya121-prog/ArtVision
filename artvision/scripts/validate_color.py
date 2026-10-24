import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.color_palette import ColorPaletteExtractor


def create_contact_sheet(sample_rows, extractor, output_path: str):
    """
    Renders a contact sheet grid showing each artwork next to its extracted palette swatches and labels.
    """
    n_images = len(sample_rows)
    fig, axes = plt.subplots(n_images, 2, figsize=(12, n_images * 2.2), gridspec_kw={"width_ratios": [1, 2]})
    if n_images == 1:
        axes = np.array([axes])

    questionable_cases = []

    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        img_path = str(row["filepath"])
        if not os.path.exists(img_path):
            img_path = os.path.join("data/processed/images", f"{row['image_id']}.jpg")

        img = Image.open(img_path).convert("RGB")
        res = extractor.analyze_image(img)

        # Left: Original Image
        ax_img = axes[idx, 0]
        ax_img.imshow(img)
        title_style = row.get("style", "Unknown")
        ax_img.set_title(f"{title_style} ({row['image_id'][:16]})", fontsize=9, fontweight="bold")
        ax_img.axis("off")

        # Right: Palette Swatches & Info
        ax_pal = axes[idx, 1]
        ax_pal.set_xlim(0, 1)
        ax_pal.set_ylim(0, 1)
        ax_pal.axis("off")

        palette = res["palette"]
        n_colors = len(palette)
        swatch_width = 0.9 / max(1, n_colors)

        for i, c in enumerate(palette):
            x = 0.05 + i * swatch_width
            # Draw color rectangle
            rect = patches.Rectangle(
                (x, 0.45), swatch_width * 0.85, 0.45,
                facecolor=c["hex"], edgecolor="black", linewidth=1.0
            )
            ax_pal.add_patch(rect)
            # Hex text & percentage
            ax_pal.text(
                x + swatch_width * 0.42, 0.32,
                f"{c['hex']}\n{c['percent']*100:.1f}%",
                ha="center", va="top", fontsize=7.5, family="monospace"
            )

        # Overall Classification Banner
        temp_color = "darkred" if res["temperature"] == "warm" else ("navy" if res["temperature"] == "cool" else "dimgray")
        info_str = (
            f"Temperature: {res['temperature'].upper()} (Warm: {res['warm_score']:.2f}, Cool: {res['cool_score']:.2f}) | "
            f"Tone: {res['brightness']} / {res['saturation']} | Pastel: {res['is_pastel']}"
        )
        ax_pal.text(0.05, 0.08, info_str, fontsize=8.5, fontweight="bold", color=temp_color)

        # Flag questionable cases (e.g., near 50/50 balance or high neutral mass)
        if abs(res["warm_score"] - res["cool_score"]) < 0.15 and res["temperature"] != "monochrome":
            questionable_cases.append({
                "image_id": str(row["image_id"]),
                "style": title_style,
                "temperature": res["temperature"],
                "warm_score": res["warm_score"],
                "cool_score": res["cool_score"],
                "reason": "Borderline balance between warm and cool tones (difference < 0.15)"
            })

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved contact sheet to {output_path}")
    return questionable_cases


def sensitivity_analysis(sample_rows, extractor, output_path: str):
    """
    Compares extracted palettes for k=3, k=5, k=8 on sample images.
    """
    k_values = [3, 5, 8]
    n_images = len(sample_rows)
    fig, axes = plt.subplots(n_images, 4, figsize=(16, n_images * 2.5), gridspec_kw={"width_ratios": [1.2, 2, 2, 2]})

    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        img_path = str(row["filepath"])
        if not os.path.exists(img_path):
            img_path = os.path.join("data/processed/images", f"{row['image_id']}.jpg")

        img = Image.open(img_path).convert("RGB")

        # Col 0: Image
        ax_img = axes[idx, 0]
        ax_img.imshow(img)
        ax_img.set_title(f"{row.get('style', '')}\n{row['image_id'][:16]}", fontsize=9)
        ax_img.axis("off")

        # Col 1, 2, 3: k=3, 5, 8
        for col_idx, k in enumerate(k_values):
            ax_k = axes[idx, col_idx + 1]
            ax_k.set_xlim(0, 1)
            ax_k.set_ylim(0, 1)
            ax_k.axis("off")

            res_k = extractor.analyze_image(img, k=k)
            palette = res_k["palette"]
            n_colors = len(palette)
            swatch_width = 0.95 / max(1, n_colors)

            for i, c in enumerate(palette):
                x = 0.02 + i * swatch_width
                rect = patches.Rectangle(
                    (x, 0.4), swatch_width * 0.88, 0.5,
                    facecolor=c["hex"], edgecolor="black", linewidth=0.8
                )
                ax_k.add_patch(rect)
                ax_k.text(
                    x + swatch_width * 0.44, 0.28,
                    f"{c['hex']}\n{c['percent']*100:.1f}%",
                    ha="center", va="top", fontsize=7, family="monospace"
                )

            label_str = f"k={k} -> {res_k['color_count']} colors ({res_k['temperature']}, W:{res_k['warm_score']:.2f}, C:{res_k['cool_score']:.2f})"
            ax_k.text(0.02, 0.05, label_str, fontsize=8, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved sensitivity analysis to {output_path}")


def speed_benchmark(df, extractor, n_samples: int = 50):
    """
    Measures CPU latency across real artworks.
    """
    sample_df = df.sample(min(n_samples, len(df)), random_state=42)
    latencies = []

    for _, row in sample_df.iterrows():
        img_path = str(row["filepath"])
        if not os.path.exists(img_path):
            img_path = os.path.join("data/processed/images", f"{row['image_id']}.jpg")

        img = Image.open(img_path)
        t0 = time.time()
        _ = extractor.analyze_image(img)
        latencies.append((time.time() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    median_ms = float(np.median(latencies))
    min_ms = float(np.min(latencies))
    max_ms = float(np.max(latencies))
    p95_ms = float(np.percentile(latencies, 95))

    print(f"\n--- CPU Speed Benchmark ({len(latencies)} images) ---")
    print(f"Mean Latency:   {mean_ms:.2f} ms")
    print(f"Median Latency: {median_ms:.2f} ms")
    print(f"Min Latency:    {min_ms:.2f} ms")
    print(f"Max Latency:    {max_ms:.2f} ms")
    print(f"P95 Latency:    {p95_ms:.2f} ms")

    return {
        "n_samples": len(latencies),
        "mean_ms": round(mean_ms, 2),
        "median_ms": round(median_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "p95_ms": round(p95_ms, 2)
    }


def style_color_eda(df, extractor, output_path: str):
    """
    Computes average warm_score and brightness across art styles and generates an EDA plot.
    """
    print("\nRunning Style Color EDA across dataset...")
    # Sample up to 30 images per style for quick and accurate aggregation
    styles = df["style"].dropna().unique().tolist()
    records = []

    for s in styles:
        sub_df = df[df["style"] == s].head(25)
        for _, row in sub_df.iterrows():
            img_path = str(row["filepath"])
            if not os.path.exists(img_path):
                img_path = os.path.join("data/processed/images", f"{row['image_id']}.jpg")

            try:
                img = Image.open(img_path)
                res = extractor.analyze_image(img)
                # Compute average lightness across palette
                avg_l = sum(c["rgb"][0]*0.299 + c["rgb"][1]*0.587 + c["rgb"][2]*0.114 for c in res["palette"]) / 255.0 if res["palette"] else 0.5
                records.append({
                    "style": s,
                    "warm_score": res["warm_score"],
                    "cool_score": res["cool_score"],
                    "temperature": res["temperature"],
                    "lightness": avg_l
                })
            except Exception:
                pass

    eda_df = pd.DataFrame(records)
    style_agg = eda_df.groupby("style").agg(
        mean_warm=("warm_score", "mean"),
        mean_cool=("cool_score", "mean"),
        mean_lightness=("lightness", "mean"),
        count=("warm_score", "count")
    ).reset_index().sort_values("mean_warm", ascending=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Warm vs Cool by Style
    x = np.arange(len(style_agg))
    w = 0.38
    ax1.bar(x - w/2, style_agg["mean_warm"], width=w, label="Mean Warm Score", color="#e76f51", edgecolor="black")
    ax1.bar(x + w/2, style_agg["mean_cool"], width=w, label="Mean Cool Score", color="#2a9d8f", edgecolor="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(style_agg["style"], rotation=45, ha="right")
    ax1.set_ylabel("Score (0.0 to 1.0)")
    ax1.set_title("Color Temperature (Warm vs Cool) by Art Style")
    ax1.legend()
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    # Plot 2: Lightness / Brightness by Style
    ax2.bar(style_agg["style"], style_agg["mean_lightness"], color="#e9c46a", edgecolor="black")
    ax2.set_xticks(x)
    ax2.set_xticklabels(style_agg["style"], rotation=45, ha="right")
    ax2.set_ylabel("Mean Luminance (0.0 = Dark, 1.0 = Light)")
    ax2.set_title("Average Artwork Luminance by Style")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved style color EDA plot to {output_path}")

    return style_agg.to_dict(orient="records")


def main():
    extractor = ColorPaletteExtractor("configs/color.yaml")
    df = pd.read_csv("data/processed/metadata.csv")

    output_dir = "runs/color_validation"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Contact Sheet of 40 diverse artworks (4 per style across 10 styles)
    styles = [
        "Baroque", "Impressionism", "Post_Impressionism", "High_Renaissance",
        "Expressionism", "Romanticism", "Realism", "Art_Nouveau", "Symbolism", "Ukiyo_e"
    ]
    sample_40 = []
    for s in styles:
        s_df = df[df["style"] == s]
        if len(s_df) >= 4:
            sample_40.append(s_df.head(4))
        else:
            sample_40.append(s_df)
    contact_df = pd.concat(sample_40).reset_index(drop=True)

    print(f"Generating contact sheet for {len(contact_df)} artworks...")
    questionable = create_contact_sheet(contact_df, extractor, os.path.join(output_dir, "contact_sheet.png"))

    # 2. Sensitivity Analysis (5 diverse artworks)
    sensitivity_df = df.groupby("style").first().loc[styles[:5]].reset_index()
    sensitivity_analysis(sensitivity_df, extractor, os.path.join(output_dir, "k_sensitivity.png"))

    # 3. Speed Benchmark
    speed_stats = speed_benchmark(df, extractor, n_samples=50)

    # 4. Style Color EDA
    eda_stats = style_color_eda(df, extractor, os.path.join(output_dir, "eda_style_color.png"))

    # Save summary report
    summary = {
        "benchmark": speed_stats,
        "questionable_cases": questionable,
        "style_eda": eda_stats
    }
    with open(os.path.join(output_dir, "color_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nAll validation tasks completed. Summary saved to {output_dir}/color_summary.json")


if __name__ == "__main__":
    main()
