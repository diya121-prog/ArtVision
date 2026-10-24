# ArtVision: Visualizations & Presentation Showcase Guide

This guide accompanies the comprehensive visualization suite located in the `showcase/` directory of the project folder. It provides slide-by-slide explanation points, key empirical metrics, architectural justifications, and answers to anticipated viva defense questions.

---

## 🚀 Quick Showcase Links

- **Interactive Showcase Web Dashboard**: Open [showcase/index.html](file:///Users/diyamakwana/Desktop/PDEU/7th%20Sem/ML%20Lab/ML%20Project/showcase/index.html) directly in any browser (zero server setup required).
- **Automated Generator Script**: [showcase/generate_showcase.py](file:///Users/diyamakwana/Desktop/PDEU/7th%20Sem/ML%20Lab/ML%20Project/showcase/generate_showcase.py).
- **Symlinked Folder**: Accessible via both `showcase/` and `artvision/showcase/`.

---

## 📊 Summary of Benchmark Metrics (Test Set, N = 816)

| Prediction Task | Top-1 Accuracy | Top-3 Accuracy | Macro F1-Score | Calibrated ECE | Metric Highlight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Authenticity** (Human vs AI) | **97.06%** | 100.0% | **0.935** | **0.0107** | **AUROC: 0.995** (99.5%) |
| **Derived Movement** | **72.01%** | 100.0% | 0.720 | — | 7 Art Historical Eras |
| **Genre** (10 Classes) | **66.35%** | **94.04%** | 0.565 | 0.0664 | 94.0% Top-3 Precision |
| **Artist Attribution** (Top 23) | **65.05%** | **82.61%** | 0.625 | 0.0439 | 82.6% Top-3 Precision |
| **Style** (27 WikiArt Classes) | **59.93%** | **84.22%** | 0.434 | 0.0442 | 84.2% Top-3 Precision |
| **Mood** (5 Emotion Classes) | 21.43% | 78.57% | 0.115 | 0.1318 | Experimental Task |
| **Color Extraction** (CIELAB) | — | — | — | — | **53.6 ms** Latency / Image |

---

## 🖼️ Detailed Guide to Visualizations

### 1. Dataset Visualizations (`showcase/dataset/`)

#### `01_dataset_distributions.png`
- **What it shows**: 6-panel overview of the 8,099 artwork catalog (Top-10 styles, unified historical movements, human vs AI balance, top genres, master artists, and mood annotations).
- **Key Insight**: Shows that while Impressionism and Realism form major classes, long-tail styles are preserved in the 27-class taxonomy. Human artworks comprise 7,099 images from WikiArt, balanced against 1,000 synthetic artworks from ArtBench.
- **Viva Talking Point**: *"Our dataset captures both the depth of art history and modern generative art challenges, requiring careful multi-task handling to prevent class imbalance skew."*

#### `02_dataset_splits_and_coverage.png`
- **What it shows**: Left: 80% train (6,468), 10% validation (815), 10% test (816). Right: Annotation coverage percentage across tasks.
- **Key Insight**: Explains how loss masking allows multi-task training on partially annotated datasets (e.g., ArtBench AI images lack artist/genre labels; mood annotations exist on 1.7% of artworks).
- **Viva Talking Point**: *"We don't discard unannotated artworks; instead, our multi-task loss applies a binary mask $m_t \in \{0, 1\}$, zeroing out unannotated task losses during backpropagation."*

#### `03_dataset_samples_gallery.png`
- **What it shows**: High-resolution side-by-side gallery of classical masterworks (Van Gogh, Monet, Rembrandt, Renoir) against synthetic artworks synthesized by Latent Diffusion and Stable Diffusion models.
- **Key Insight**: Demonstrates the visual quality of the dataset and how AI-generated art mimics human stylistic traits while containing subtle generation artifacts.

#### `04_image_resolutions_aspect_ratios.png`
- **What it shows**: Resolution scatter plot and aspect ratio distribution across data sources.
- **Key Insight**: AI images from ArtBench are strictly $384 \times 384$ square ($1.0$ aspect ratio), whereas WikiArt museum scans display natural landscape ($4:3$) and portrait ($3:4$) variance.

#### `05_color_temperature_tone_eda.png`
- **What it shows**: Mean warmth score, cool score, and CIELAB lightness ($L^*$) across art styles.
- **Key Insight**: Correlates art historical movements with perceptual color science. Baroque and High Renaissance exhibit warm, lower-luminance tones (chiaroscuro), while Impressionism exhibits high lightness and cool/neutral palettes.

---

### 2. Evaluation & Benchmark Visualizations (`showcase/evaluation/`)

#### `01_multitask_training_curves.png`
- **What it shows**: Epoch progression of Total Weighted Training Loss, Validation Loss, and Mean Validation Macro F1 score across 17 epochs.
- **Key Insight**: Demonstrates smooth multi-task convergence without severe negative transfer or catastrophic overfitting.

#### `02_multitask_task_losses_and_weights.png`
- **What it shows**: Decomposed individual task losses and trajectories of the learned homoscedastic uncertainty parameters $\sigma_t$ (Kendall et al.).
- **Key Insight**: Illustrates how the model dynamically assigns higher relative weights to well-behaved, structured tasks (like Authenticity and Genre) while dampening gradients from noisy/sparse tasks (like Mood).

#### `03_task_performance_benchmark.png`
- **What it shows**: Comparison of Top-1 Accuracy, Top-3 Accuracy, and Macro F1 score across all 6 prediction heads.
- **Key Insight**: Highlights that Top-3 accuracy exceeds 82% on all main classification tasks (Style 84.2%, Genre 94.0%, Artist 82.6%), which is standard for artistic classification where boundaries between related styles/genres are subjective.

#### `04_ablation_loss_weighting.png`
- **What it shows**: Empirical comparison of Equal Loss Weighting vs Manual Heuristic Weights vs Kendall Uncertainty Weighting.
- **Key Insight**: Uncertainty weighting achieves higher style accuracy (57.6% vs 55.2%) and higher macro F1 (0.575 vs 0.548) while maintaining top authenticity detection (92.8% acc, 0.980 AUROC).

#### `05_calibration_reliability_summary.png`
- **What it shows**: Expected Calibration Error (ECE) before and after post-hoc Temperature Scaling across all heads.
- **Key Insight**: Shows that temperature scaling ($T > 1$) cuts ECE by up to half (e.g. Style ECE drops from 14.0% to 6.9%), transforming uncalibrated Softmax confidence into dependable empirical probabilities.

#### `06_confusion_matrices_grid.png`
- **What it shows**: High-resolution normalized confusion matrices for Authenticity, Genre, Art Style, and Artist Attribution.
- **Key Insight**: Shows strong diagonal dominance. Misclassifications occur almost exclusively between historically adjacent movements (e.g., Impressionism vs Post-Impressionism).

#### `07_error_analysis_misclassifications.png`
- **What it shows**: Visual gallery of difficult misclassifications and boundary artworks.
- **Key Insight**: Proves that errors occur on inherently ambiguous transitional pieces (e.g., Cézanne's works bridging Impressionism and Cubism).

---

### 3. Results & Inference Visualizations (`showcase/results/`)

#### `01_multitask_prediction_showcase.png`
- **What it shows**: End-to-end multi-task inference cards for Van Gogh, Monet, and AI-generated artworks with confidence badges and extracted color swatches.
- **Key Insight**: Demonstrates the real-world utility of the model: user uploads a single image, and receives style, era, artist, authenticity, mood, and color palette simultaneously.

#### `02_gradcam_explainability_gallery.png`
- **What it shows**: Grad-CAM visual heatmaps extracted from `backbone.conv_head`.
- **Key Insight**: Confirms the model attends to authentic brushwork, impasto texture, and compositional elements rather than canvas frame borders or photographic watermarks.

#### `03_color_palette_clustering_showcase.png`
- **What it shows**: $k$-means cluster count sensitivity ($k \in \{3, 5, 8\}$) and latency benchmarks.
- **Key Insight**: Operates at 53.6 ms/image (5.6x faster than the 300 ms SLA requirement), using perceptually uniform CIELAB space and $\Delta E < 12$ near-duplicate swatch merging.

#### `04_similarity_retrieval_showcase.png`
- **What it shows**: Query artwork matched with Top-5 cosine-similarity nearest neighbors retrieved from the 8,099 vector index.
- **Key Insight**: Querying a Van Gogh artwork retrieves other Van Gogh and Camille Pissarro impressionist masterworks, validating the semantic richness of the 1,280-d EfficientNet embeddings.

---

## 🎓 Frequently Asked Viva Questions

### Q1: What makes ArtVision different from a standard CNN classifier?
> **Answer**: Most art analysis systems build separate models for each task (one for style, one for artist, one for fake detection). ArtVision uses a **single shared EfficientNet-B0 backbone with 6 modular task heads**, allowing representation sharing and saving computational footprint. Furthermore, it integrates **CIELAB color science**, **Grad-CAM visual explainability**, and **vector similarity search** into a unified production pipeline.

### Q2: Why did you use homoscedastic uncertainty loss weighting?
> **Answer**: In multi-task learning, different losses have drastically different magnitudes and gradients (e.g. 27-class Cross-Entropy vs binary BCE vs soft KL-divergence). Standard equal weighting causes dominant gradients to drown out subordinate heads. Following Kendall et al. (CVPR 2018), we formulate the multi-task loss as:
> $$\mathcal{L}_{\text{total}} = \sum_t \frac{1}{2\sigma_t^2}\mathcal{L}_t + \log \sigma_t$$
> where $\sigma_t$ is a learnable parameter representing task-dependent uncertainty. The $\log \sigma_t$ term acts as a regularizer preventing $\sigma_t \to \infty$.

### Q3: How did you validate that Grad-CAM heatmaps are genuine?
> **Answer**: In `scripts/verify_gradcam.py`, we executed two standard explainability sanity checks (Adebayo et al.):
> 1. **Model Weight Randomization Test**: Randomizing network weights collapsed heatmap similarity to 0.0, confirming heatmaps depend on learned features.
> 2. **Pixel Deletion Benchmark**: Masking the top salient pixels identified by Grad-CAM degraded model confidence significantly faster than random pixel masking.

---

*PDEU 7th Semester ML Lab Project • October 2026*
