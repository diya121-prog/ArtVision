# ArtVision

An Explainable Multi-Task Deep Learning Framework for Comprehensive Artwork Analysis.

## Overview
User uploads one artwork image. The system returns:
- Art style, art movement, artist, era, genre/subject, mood, technique
- Dominant color palette (warm / cool / monochrome)
- Authenticity (human-made vs AI-generated) with confidence
- Top-5 visually similar artworks
- Grad-CAM heatmap explaining which regions drove the prediction

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Phase 3: Style Classification (Single-task)
We use an EfficientNet-B0 backbone for single-task style classification.

### Training
To train the model on the human-made dataset:
```bash
python3 -m src.training.train --config configs/train_style.yaml
```
Metrics, logs, and checkpoints are saved to `runs/style_model/`.

### Evaluation
To evaluate the model on both human and AI test splits and generate misclassification grids:
```bash
python3 src/training/evaluate.py --config configs/train_style.yaml --checkpoint runs/style_model/best_model.pth
```
This generates the confusion matrix, top confused pairs, and calibration metrics.

### Inference
To predict the style of a single image:
```bash
python3 scripts/predict_style.py --image path/to/image.jpg --checkpoint runs/style_model/best_model.pth
```

## Phase 4: Multi-Task Artwork Analysis Model
A shared EfficientNet-B0 backbone with modular task heads jointly predicting:
- **Style** (10 classes, softmax classification)
- **Derived Movement** (Post-hoc art historical mapping from style)
- **Artist** (Top-N artists with min count filtering; all others masked)
- **Genre** (Config-driven head; graceful 0-loss handling when unannotated)
- **Mood** (5-class distribution with KL-divergence on soft distribution / CE majority)
- **Authenticity** (Binary logit for human-made vs AI-generated detection)
- **Era & Technique** (Extensible disabled placeholders)

### Training
Train the multi-task model with uncertainty loss weighting (Kendall et al.):
```bash
python3 -m src.training.train_multitask --config configs/train_multitask.yaml
```

To run ablation experiments across loss weighting schemes:
```bash
python3 -m src.training.train_multitask --weighting equal --output_dir runs/ablation_equal
python3 -m src.training.train_multitask --weighting manual --output_dir runs/ablation_manual
python3 -m src.training.train_multitask --weighting uncertainty --output_dir runs/multitask_model
```

### Evaluation & Calibration
Evaluate on the test set and fit temperature scaling per head on the validation set:
```bash
python3 -m src.training.evaluate_multitask --checkpoint runs/multitask_model/best_model.pth
```
This saves:
- `test_metrics.json`
- `calibration.json`
- Confusion matrices (`confusion_matrix_<task>.png`)
- Reliability diagrams before and after temperature scaling (`calibration_<task>_*.png`)
- Misclassification grids (`misclassifications_<task>.png`)

### Inference
Run calibrated multi-task prediction on an artwork image:
```bash
python3 predict.py --image path/to/image.jpg --checkpoint runs/multitask_model/best_model.pth
```
Outputs calibrated JSON schema with confidence scores and derived movement.

## Phase 5: Color Palette & Tone Analysis
A lightweight, fast, non-neural color analysis module based on K-Means clustering in the perceptually uniform CIELAB color space (`cv2.COLOR_RGB2LAB`), near-duplicate cluster merging ($\Delta E_{76} < 12.0$), and rule-based temperature and tone classification.

### Features
- **CIELAB K-Means Clustering**: Perceptually uniform color clustering preventing RGB Euclidean distortion of chromatic distances.
- **Near-Duplicate Swatch Merging**: Merges clusters with $\Delta E < 12.0$ into weighted centroids to avoid dividing gradients into duplicate swatches.
- **Temperature Classification**: Robust chroma-weighted voting classifying into `warm`, `cool`, `neutral-mixed`, or `monochrome`, reporting continuous `warm_score` and `cool_score` in $[0, 1]$.
- **Tone Descriptors**: Rule-based categorization of `brightness` (`dark`, `mid`, `light`), `saturation` (`muted`, `balanced`, `vivid`), and `is_pastel` (boolean flag).
- **Edge-Case Resilience**: Handles RGBA transparency (composited onto white), grayscale, solid single-color images, and tiny inputs without division-by-zero or crashes.
- **Blazing Performance**: ~53 ms CPU latency per image on Apple M4 (over 5.6x faster than the 300 ms target).

### Configuration
All hyperparameters are configurable in `configs/color.yaml`:
- `palette.k_clusters`: Default cluster count (default: 5)
- `palette.max_side`: Downscale longest side in pixels (default: 180)
- `palette.merge_delta_e_threshold`: CIELAB $\Delta E$ distance for duplicate merging (default: 12.0)
- `temperature`: Warm/cool hue ranges, chroma thresholds, monochrome rules
- `tone`: Brightness, saturation, and pastel thresholds

### Validation & Benchmark
Run the comprehensive test suite and validation scripts:
```bash
python tests/test_color_palette.py
python scripts/validate_color.py
```
This generates:
- `runs/color_validation/contact_sheet.png`: 40 artworks with swatches & temperature labels
- `runs/color_validation/k_sensitivity.png`: Comparison of $k \in \{3, 5, 8\}$ across artworks
- `runs/color_validation/eda_style_color.png`: Mean warmth and luminance across art styles
- `runs/color_validation/color_summary.json`: Speed metrics and EDA summary

### Integrated Prediction
Color palette extraction is seamlessly integrated into `predict.py`:
```bash
python predict.py --image path/to/image.jpg
```
The output JSON includes the full `"color"` block with swatches (hex, RGB, percentage), temperature classification, continuous warm/cool scores, and tone descriptors alongside the multi-task model predictions.

## Phase 6: Grad-CAM Visual Explainability
Hook-based Grad-CAM implementation (`src/explain/gradcam.py`) targeting the last convolutional layer of the EfficientNet-B0 backbone (`backbone.conv_head`).

### Capabilities
- **Context-Managed Hooks**: Forward and backward hooks automatically register and unregister cleanly without memory leaks.
- **Multi-Task Head Support**: Generates spatial heatmaps for any head (`style`, `artist`, `genre`, `mood`, `authenticity`).
- **Visual Overlay**: Alpha-blended Jet colormap superimposed on original images.
- **Border-Mass Metric**: Quantifies perimeter attribution to detect framing/border artifacts.

### CLI Usage
Generate prediction with visual explanation overlay:
```bash
python predict.py --image path/to/artwork.jpg --explain --explain-task style
```
Saves overlay to `runs/explain/<image_name>_<task>_gradcam.jpg` and reports `border_mass_fraction`.

### Sanity Checks
Run the explanation verification suite:
```bash
python scripts/verify_gradcam.py
```
- **Randomization Test**: Verifies heatmap cosine similarity collapses to 0.0 when model weights are randomized.
- **Deletion Test**: Confirms salient pixel deletion degrades target prediction confidence faster than random pixel masking.
- **Border-Mass Fraction**: Confirms attribution concentrates on artwork subjects rather than edge borders.

## Phase 7: Artwork Retrieval & Similarity Search
Lightweight cosine-similarity retrieval system operating on L2-normalized 1280-dimensional EfficientNet embeddings across all 8,099 catalog artworks.

### Building / Rebuilding Embeddings
To re-extract embeddings from `metadata.csv` (never unindexed files):
```bash
python src/features/build_embeddings.py
```
Saves:
- `runs/retrieval/embeddings.npy` (Shape: `[8099, 1280]`, float32, unit norm)
- `runs/retrieval/image_ids.json` (Aligned metadata records)

### Querying Similar Artworks
```python
from src.features.search import ArtworkSearcher

searcher = ArtworkSearcher()
# Query by catalog ID or image path:
results = searcher.find_similar("wikiart_001797", top_k=5)
```
- **Self-Exclusion**: A queried catalog ID is never returned in its own results.
- **Near-Duplicate Filter**: Automatically filters duplicate scans (`cosine similarity > 0.995`).

## FastAPI Backend Server
Production-ready FastAPI application (`api/main.py`) serving prediction, explainability, similarity search, and static assets.

### Startup
Run the server with Uvicorn:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints
1. `GET /health`: Server status, active device (`mps`/`cpu`), loaded checkpoint, and enabled task heads.
2. `POST /predict`: Multipart or raw binary image upload.
   - Query params: `explain` (bool), `explain_task` (str).
   - Returns full analysis matching `predict.py` schema with `schema_version: "1.0"`.
3. `POST /similar`: Artwork retrieval.
   - Accepts multipart image upload or JSON payload `{"image_id": "..."}`.
   - Query param: `top_k` (int, default: 5).
4. `GET /static/images/<id>.jpg`: Static artwork thumbnails.
5. `GET /static/heatmaps/<filename>.jpg`: Generated Grad-CAM overlay heatmaps.

### API Contract & Class Taxonomy
- **Style Taxonomy**: Predictions strictly output canonical raw 27-class WikiArt labels (e.g. `Synthetic_Cubism`, `Analytical_Cubism`, `Realism`) to preserve fine-grained historical classification.
- **Movement Grouping**: The `movement` field derives the artwork's art historical era using a unified 7-group coarse taxonomy:
  1. *Renaissance & Mannerism*
  2. *Baroque & Rococo*
  3. *19th-Century Realism & Romanticism*
  4. *Late 19th-Century Avant-Garde*
  5. *Early 20th-Century Modernism* (includes `Naive_Art_Primitivism`)
  6. *Post-War & Contemporary Art*
  7. *Traditional East Asian Art*
- **Authenticity Caveat**: The authenticity block outputs binary classification (`human_made` vs `ai_generated`) and explicitly includes:
  `"caveat": "Trained on specific generators; less reliable on resized, compressed or screenshotted images."`
- **Mood Task**: Flagged with `"experimental": true`.

### Checkpoint Swap
To deploy a new model checkpoint:
1. Update `model.pretrained_checkpoint` or point `--checkpoint` to your new `.pth` file.
2. Re-derive validation calibration parameters: `python src/training/calibration.py`.
3. Rebuild retrieval embeddings: `python src/features/build_embeddings.py`.
4. Restart the API server.

