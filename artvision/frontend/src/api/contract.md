# ArtVision API Contract: `POST /predict` & `POST /similar`

- **Schema Version**: `1.0` (compatible with `1.0.0`)
- **Base URL**: `http://localhost:8000` (configurable via `VITE_API_URL`)
- **Prediction Endpoint**: `POST /predict?explain={true|false}&explain_task={style|artist|genre|mood|authenticity}`
- **Similarity Search Endpoint**: `POST /similar?top_k=5`
- **Static Asset Mounts**:
  - `/static/images/<id>.jpg` (Artwork thumbnails)
  - `/static/heatmaps/<filename>.jpg` (Grad-CAM overlays)

---

## 1. Overview & Coarse Taxonomies

The ArtVision backend provides unified multi-task artwork analysis, visual explainability via hook-based Grad-CAM, and vector similarity retrieval over 8,099 catalog artworks using L2-normalized EfficientNet embeddings.

### 1.1. Movement Coarse Period Taxonomy
The `movement` field derives the artwork's historical epoch from the predicted fine-grained style across 7 coarse groups:
1. **Renaissance & Mannerism** (Early/High/Northern Renaissance, Mannerism)
2. **Baroque & Rococo** (Baroque, Rococo)
3. **19th-Century Realism & Romanticism** (Realism, Romanticism)
4. **Late 19th-Century Avant-Garde** (Impressionism, Post-Impressionism, Symbolism, Art Nouveau)
5. **Early 20th-Century Modernism** (Fauvism, Expressionism, Cubism, Art Deco, Naive Art)
6. **Post-War & Contemporary Art** (Abstract Expressionism, Pop Art, Color Field)
7. **Traditional East Asian Art** (Ukiyo-e)

### 1.2. Artist Head Representation
Trained on 23 cataloged master artists with min-count filtering. All unrepresented artists or AI works are masked to `null` or marked `uncertain: true`.

### 1.3. Mood Head & Subjectivity
Flagged with `"experimental": true`. Soft KL-divergence calibrated distribution over 5 emotional categories.

### 1.4. Authenticity Caveat
Binary classification (`is_ai_generated: boolean`, `label: "ai_generated" | "human_made"`, `probability_ai: number`).
Always includes:
`"caveat": "Trained on specific generators; less reliable on resized, compressed or screenshotted images."`

---

## 2. API Endpoints

### 2.1. `POST /predict`
- **Query Params**:
  - `explain`: boolean (default `false`)
  - `explain_task`: `"style"` | `"artist"` | `"genre"` | `"mood"` | `"authenticity"` (default `"style"`)
- **Body**: Multipart form with `image=@<filepath>` or raw binary image bytes.
- **Response**:
```json
{
  "schema_version": "1.0",
  "style": {
    "top3": [{ "style": "Realism", "confidence": 0.6103 }],
    "prediction": "Realism",
    "confidence": 0.6103,
    "uncertain": false
  },
  "movement": {
    "derived_from_style": "Realism",
    "movement": "19th-Century Realism & Romanticism"
  },
  "artist": {
    "top3": [{ "artist": "Vincent Van Gogh", "confidence": 0.9450 }],
    "prediction": "Vincent Van Gogh",
    "confidence": 0.9450,
    "uncertain": false
  },
  "genre": {
    "top3": [{ "genre": "Sketch And Study", "confidence": 0.5078 }],
    "prediction": "Sketch And Study",
    "confidence": 0.5078,
    "uncertain": true
  },
  "mood": {
    "prediction": "Joyful / Excited",
    "confidence": 0.5302,
    "distribution": { "Joyful / Excited": 0.5302, "Awe / Sublime": 0.3705 },
    "uncertain": false,
    "experimental": true
  },
  "authenticity": {
    "is_ai_generated": false,
    "label": "human_made",
    "probability_ai": 0.0001,
    "confidence": 0.9999,
    "uncertain": false,
    "caveat": "Trained on specific generators; less reliable on resized, compressed or screenshotted images."
  },
  "era": null,
  "technique": null,
  "color": {
    "palette": [{ "hex": "#ca9b68", "rgb": [202, 155, 104], "percent": 0.4732 }],
    "color_count": 5,
    "temperature": "warm",
    "warm_score": 1.0,
    "cool_score": 0.0,
    "brightness": "mid",
    "saturation": "balanced",
    "is_pastel": false
  },
  "explanation": {
    "task": "style",
    "target_prediction": "Realism",
    "overlay_path": "runs/explain/..._style_gradcam.jpg",
    "overlay_url": "/static/heatmaps/..._style_gradcam.jpg",
    "border_mass_fraction": 0.29,
    "heatmap_dimensions": [224, 224]
  }
}
```

### 2.2. `POST /similar`
- **Query Params**: `top_k=5`
- **Body**: Multipart with `image` file or JSON `{"image_id": "..."}`
- **Response**:
```json
{
  "schema_version": "1.0",
  "query": "uploaded_image",
  "count": 5,
  "results": [
    {
      "image_id": "wikiart_002242",
      "similarity": 0.4634,
      "style": "Realism",
      "artist": "Vincent Van Gogh",
      "genre": "Landscape",
      "is_ai_generated": false,
      "image_url": "/static/images/wikiart_002242.jpg"
    }
  ]
}
```
