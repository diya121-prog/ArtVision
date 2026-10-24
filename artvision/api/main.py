import os
import sys
import uuid
import json
import shutil
from typing import Optional, Dict, Any, List
from email import policy
from email.parser import BytesParser

import yaml
from PIL import Image
import torch
torch.set_num_threads(1)
from fastapi import FastAPI, Request, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from predict import predict_artwork
from src.features.search import ArtworkSearcher

app = FastAPI(
    title="ArtVision Multi-Task API",
    version="1.0.0",
    description="Multi-task neural analysis, attribution, explanation, and similarity search for artworks."
)

# Enable CORS for local development and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure required directories exist
os.makedirs("runs/explain", exist_ok=True)
os.makedirs("runs/temp", exist_ok=True)
os.makedirs("data/processed/images", exist_ok=True)

# Mount static files
app.mount("/static/images", StaticFiles(directory="data/processed/images"), name="images")
app.mount("/static/heatmaps", StaticFiles(directory="runs/explain"), name="heatmaps")

# Global searcher instance
searcher_instance: Optional[ArtworkSearcher] = None


def get_searcher() -> ArtworkSearcher:
    global searcher_instance
    if searcher_instance is None:
        searcher_instance = ArtworkSearcher(
            embeddings_path="runs/retrieval/embeddings.npy",
            image_ids_path="runs/retrieval/image_ids.json",
            checkpoint_path="runs/multitask_model/best_model.pth"
        )
    return searcher_instance


def parse_request_image_and_data(request: Request, body_bytes: bytes):
    """
    Robust extractor handling raw image uploads, multipart/form-data, and JSON.
    Uses Python standard library to ensure zero fragile binary dependencies.
    """
    content_type = request.headers.get("content-type", "")
    image_bytes = None
    form_data = {}

    if "multipart/form-data" in content_type:
        headers_blob = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8")
        msg = BytesParser(policy=policy.default).parsebytes(headers_blob + body_bytes)
        for part in msg.iter_parts():
            part_name = part.get_param("name", header="content-disposition")
            if part.get_filename() or part_name in ["file", "image"]:
                image_bytes = part.get_payload(decode=True)
            else:
                payload = part.get_payload(decode=True)
                if payload:
                    form_data[part_name] = payload.decode("utf-8", errors="ignore")
    elif content_type.startswith("image/") or content_type == "application/octet-stream":
        image_bytes = body_bytes
    elif "application/json" in content_type:
        try:
            form_data = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            pass

    return image_bytes, form_data


@app.get("/health", summary="Health Check")
def health():
    """
    Reports API operational status, active device, loaded checkpoint, and enabled multi-task heads.
    """
    import torch
    ckpt_path = "runs/multitask_model/best_model.pth"
    device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tasks_enabled = []
    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            tasks_enabled = [t for t, cfg in ckpt.get("config", {}).get("tasks", {}).items() if cfg.get("enabled", False)]
        except Exception:
            pass

    return {
        "status": "ok",
        "device": device_name,
        "checkpoint": ckpt_path,
        "tasks_enabled": tasks_enabled,
        "total_catalog_artworks": 8099,
        "version": "1.0.0"
    }


@app.post("/predict", summary="Predict Artwork Attributes & Authenticity")
async def predict_endpoint(
    request: Request,
    explain: bool = Query(False, description="Generate Grad-CAM visual heatmap overlay"),
    explain_task: str = Query("style", description="Task head to explain with Grad-CAM")
):
    """
    Analyzes an uploaded artwork image across style, artist, genre, mood, authenticity, and color.
    Returns calibrated confidences, uncertainty flags, and optional Grad-CAM visual explanation.
    """
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body. Please provide an image file.")

    if len(body) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 15MB.")

    image_bytes, _ = parse_request_image_and_data(request, body)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image found in request. Provide multipart file or raw image bytes.")

    temp_filename = f"upload_{uuid.uuid4().hex}.jpg"
    temp_path = os.path.join("runs/temp", temp_filename)

    try:
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        # Validate that the image can be decoded
        try:
            with Image.open(temp_path) as im:
                im.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format. Ensure image is JPEG, PNG, or WebP.")

        # Run multi-task prediction
        prediction = predict_artwork(
            image_path=temp_path,
            checkpoint_path="runs/multitask_model/best_model.pth",
            calibration_path="runs/multitask_model/calibration.json",
            color_config_path="configs/color.yaml",
            predict_config_path="configs/predict.yaml",
            explain=explain,
            explain_task=explain_task,
            explain_output_dir="runs/explain"
        )

        # Attach schema metadata and clean URLs
        response_dict = {
            "schema_version": "1.0",
            **prediction
        }

        # If explanation was generated, convert local path to web-accessible URL
        if response_dict.get("explanation") and "overlay_path" in response_dict["explanation"]:
            overlay_rel = os.path.basename(response_dict["explanation"]["overlay_path"])
            response_dict["explanation"]["overlay_url"] = f"/static/heatmaps/{overlay_rel}"

        return JSONResponse(content=response_dict)

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        import gc
        gc.collect()


@app.post("/similar", summary="Artwork Similarity Search")
async def similar_endpoint(
    request: Request,
    top_k: int = Query(5, ge=1, le=20, description="Number of nearest artworks to return")
):
    """
    Searches the catalog of 8,099 artworks using L2-normalized EfficientNet embeddings.
    Accepts an uploaded image file OR an existing catalog image_id.
    Enforces self-exclusion and filters near-duplicate copies (similarity > 0.995).
    """
    body = await request.body()
    image_bytes, form_data = parse_request_image_and_data(request, body)
    query_id = form_data.get("image_id") or request.query_params.get("image_id")

    searcher = get_searcher()

    if image_bytes:
        temp_filename = f"query_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join("runs/temp", temp_filename)
        try:
            with open(temp_path, "wb") as f:
                f.write(image_bytes)
            results = searcher.find_similar(temp_path, top_k=top_k)
            query_label = "uploaded_image"
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    elif query_id:
        if query_id not in searcher.id_to_idx:
            raise HTTPException(status_code=404, detail=f"Image ID '{query_id}' not found in catalog.")
        results = searcher.find_similar(query_id, top_k=top_k)
        query_label = query_id
    else:
        raise HTTPException(status_code=400, detail="Provide either an uploaded image or an 'image_id' parameter.")

    # Enhance results with web URLs for thumbnails
    for r in results:
        r["image_url"] = f"/static/images/{r['image_id']}.jpg"

    return {
        "schema_version": "1.0",
        "query": query_label,
        "count": len(results),
        "results": results
    }


# Root route with graceful fallback
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, summary="ArtVision Interactive Dashboard")
def root_index():
    frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist"))
    index_file = os.path.join(frontend_dist, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            return HTMLResponse(content=f.read())
            
    # Standalone interactive dashboard when frontend/dist is not bundled
    return HTMLResponse(content="""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ArtVision | Artwork Analysis & Authentication</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --border: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body { background-color: var(--bg); color: var(--text); padding: 2rem 1rem; line-height: 1.5; min-height: 100vh; }
    .container { max-width: 1100px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 2.5rem; }
    .badge { display: inline-block; padding: 0.35rem 0.85rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem; background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }
    h1 { font-size: 2.6rem; font-weight: 800; background: linear-gradient(135deg, #a5b4fc 0%, #ec4899 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    p.sub { color: var(--text-muted); font-size: 1.1rem; max-width: 650px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 2rem; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; padding: 1.75rem; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3); }
    .card-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }
    .upload-zone { border: 2px dashed #334155; border-radius: 12px; padding: 2.5rem 1rem; text-align: center; cursor: pointer; transition: all 0.2s; background: rgba(15, 23, 42, 0.5); }
    .upload-zone:hover, .upload-zone.dragover { border-color: var(--primary); background: rgba(99, 102, 241, 0.05); }
    .upload-zone input { display: none; }
    .preview-box { margin-top: 1rem; border-radius: 10px; overflow: hidden; display: none; max-height: 320px; text-align: center; background: #000; }
    .preview-box img { max-width: 100%; max-height: 320px; object-fit: contain; }
    .btn { display: block; width: 100%; padding: 0.85rem; border: none; border-radius: 10px; font-weight: 700; font-size: 1rem; cursor: pointer; transition: all 0.2s; background: var(--primary); color: white; margin-top: 1.25rem; }
    .btn:hover { background: var(--primary-hover); transform: translateY(-1px); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
    .select-group { margin-top: 1rem; }
    .select-group label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 600; }
    select { width: 100%; padding: 0.65rem; border-radius: 8px; background: #0f172a; border: 1px solid var(--border); color: var(--text); outline: none; }
    .result-section { display: none; }
    .auth-banner { padding: 1rem 1.25rem; border-radius: 12px; margin-bottom: 1.25rem; display: flex; justify-content: space-between; align-items: center; }
    .auth-human { background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #34d399; }
    .auth-ai { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); color: #f87171; }
    .res-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.25rem; }
    .stat-card { background: #0f172a; padding: 1rem; border-radius: 10px; border: 1px solid var(--border); }
    .stat-label { font-size: 0.75rem; text-transform: uppercase; color: var(--text-muted); font-weight: 700; letter-spacing: 0.05em; }
    .stat-value { font-size: 1.25rem; font-weight: 700; margin-top: 0.25rem; color: #fff; }
    .palette-row { display: flex; gap: 8px; margin: 1rem 0; }
    .swatch { flex: 1; height: 50px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); position: relative; }
    .cam-box { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 1rem; }
    .cam-img-wrap { border-radius: 8px; overflow: hidden; background: #000; text-align: center; }
    .cam-img-wrap img { width: 100%; height: 180px; object-fit: contain; }
    .cam-img-wrap p { font-size: 0.75rem; color: var(--text-muted); padding: 0.35rem 0; }
    .spinner { display: none; text-align: center; padding: 3rem 1rem; }
    .loading-dot { display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: var(--primary); margin: 0 4px; animation: bounce 0.6s infinite alternate; }
    .loading-dot:nth-child(2) { animation-delay: 0.2s; }
    .loading-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes bounce { to { opacity: 0.3; transform: translateY(-8px); } }
    footer { text-align: center; margin-top: 3rem; color: var(--text-muted); font-size: 0.85rem; }
    footer a { color: #818cf8; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <span class="badge">Multi-Task Deep Learning</span>
      <h1>ArtVision</h1>
      <p class="sub">Explainable neural analysis, authenticity verification (Human vs AI), color theory extraction, and Grad-CAM interpretability.</p>
    </header>

    <div class="grid">
      <!-- Left Column: Input -->
      <div class="card">
        <div class="card-title"><span>🎨</span> Select Artwork</div>
        <div class="upload-zone" id="dropZone">
          <p style="font-size: 2.2rem; margin-bottom: 0.5rem;">🖼️</p>
          <p style="font-weight: 600; margin-bottom: 0.25rem;">Click or drag artwork here</p>
          <p style="font-size: 0.85rem; color: var(--text-muted);">Supports JPEG, PNG, WebP (up to 15MB)</p>
          <input type="file" id="fileInput" accept="image/*">
        </div>

        <div class="preview-box" id="previewBox">
          <img id="previewImg" src="" alt="Preview">
        </div>

        <div class="select-group">
          <label for="explainTask">Grad-CAM Heatmap Target:</label>
          <select id="explainTask">
            <option value="style" selected>Style Classification Head</option>
            <option value="authenticity">Authenticity Head (Human vs AI)</option>
            <option value="artist">Artist Attribution Head</option>
            <option value="genre">Genre / Subject Head</option>
            <option value="mood">Affective Mood Head</option>
          </select>
        </div>

        <button class="btn" id="analyzeBtn" disabled>🚀 Run Multi-Task Analysis</button>
      </div>

      <!-- Right Column: Results -->
      <div class="card">
        <div class="card-title"><span>📊</span> Neural Attribution Results</div>

        <div class="spinner" id="loadingSpinner">
          <div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div>
          <p style="margin-top: 1rem; color: var(--text-muted); font-size: 0.9rem;">Executing calibrated inference & generating Grad-CAM heatmaps...</p>
        </div>

        <div id="placeholder" style="text-align: center; padding: 4rem 1rem; color: var(--text-muted);">
          <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">🔍</p>
          <p>Upload an artwork on the left to inspect its art movement, painter attribution, authenticity, and visual heatmaps.</p>
        </div>

        <div class="result-section" id="resultsSection">
          <div class="auth-banner" id="authBanner">
            <div>
              <div style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase;" id="authTag">Authenticity</div>
              <div style="font-size: 1.25rem; font-weight: 800;" id="authVerdict">Human Made Masterwork</div>
            </div>
            <div style="font-size: 1.3rem; font-weight: 800;" id="authScore">98.5%</div>
          </div>

          <div class="res-grid">
            <div class="stat-card">
              <div class="stat-label">Style & Era</div>
              <div class="stat-value" id="styleVal">-</div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;" id="movementVal">-</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Artist Attribution</div>
              <div class="stat-value" id="artistVal">-</div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;" id="artistConf">-</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Genre / Subject</div>
              <div class="stat-value" id="genreVal">-</div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;" id="genreConf">-</div>
            </div>
            <div class="stat-card">
              <div class="stat-label">Color & Tone</div>
              <div class="stat-value" id="colorTemp">-</div>
              <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;" id="colorTone">-</div>
            </div>
          </div>

          <div style="margin-bottom: 1.25rem;">
            <div class="stat-label">CIELAB Dominant Swatches</div>
            <div class="palette-row" id="paletteRow"></div>
          </div>

          <div id="camSection" style="display: none;">
            <div class="stat-label">Grad-CAM Interpretability Overlay</div>
            <div class="cam-box">
              <div class="cam-img-wrap">
                <img id="camOriginal" src="" alt="Original">
                <p>Input Artwork</p>
              </div>
              <div class="cam-img-wrap">
                <img id="camOverlay" src="" alt="Grad-CAM Overlay">
                <p id="camCaption">Heatmap Overlay</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <footer>
      <p>ArtVision Framework • PDEU ML Project • <a href="/docs">Interactive API Docs (Swagger)</a> • <a href="/health">System Health</a></p>
    </footer>
  </div>

  <script>
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const previewBox = document.getElementById('previewBox');
    const previewImg = document.getElementById('previewImg');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const placeholder = document.getElementById('placeholder');
    const resultsSection = document.getElementById('resultsSection');
    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });

    function handleFile(file) {
      selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewBox.style.display = 'block';
        analyzeBtn.disabled = false;
      };
      reader.readAsDataURL(file);
    }

    analyzeBtn.addEventListener('click', async () => {
      if (!selectedFile) return;

      placeholder.style.display = 'none';
      resultsSection.style.display = 'none';
      loadingSpinner.style.display = 'block';
      analyzeBtn.disabled = true;

      const explainTask = document.getElementById('explainTask').value;
      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const response = await fetch(`/predict?explain=true&explain_task=${explainTask}`, {
          method: 'POST',
          body: formData
        });

        if (!response.ok) throw new Error('API Error: ' + response.statusText);
        const data = await response.json();

        // Render Results
        loadingSpinner.style.display = 'none';
        resultsSection.style.display = 'block';

        // Authenticity
        const auth = data.authenticity || {};
        const isAi = auth.is_ai_generated;
        const banner = document.getElementById('authBanner');
        banner.className = 'auth-banner ' + (isAi ? 'auth-ai' : 'auth-human');
        document.getElementById('authVerdict').textContent = isAi ? '⚠️ Synthetic (AI-Generated)' : '✅ Human-Made Artwork';
        document.getElementById('authScore').textContent = Math.round((auth.confidence || 0) * 100) + '%';

        // Style & Movement
        const style = data.style || {};
        document.getElementById('styleVal').textContent = style.prediction || 'Unknown';
        document.getElementById('movementVal').textContent = (data.movement && data.movement.prediction) ? ('Movement: ' + data.movement.prediction) : ('Confidence: ' + Math.round((style.confidence || 0)*100) + '%');

        // Artist
        const artist = data.artist || {};
        document.getElementById('artistVal').textContent = artist.prediction || 'Unknown';
        document.getElementById('artistConf').textContent = 'Confidence: ' + Math.round((artist.confidence || 0)*100) + '%';

        // Genre
        const genre = data.genre || {};
        document.getElementById('genreVal').textContent = genre.prediction || 'General';
        document.getElementById('genreConf').textContent = 'Confidence: ' + Math.round((genre.confidence || 0)*100) + '%';

        // Color
        const color = data.color || {};
        document.getElementById('colorTemp').textContent = (color.temperature || 'Balanced').toUpperCase();
        document.getElementById('colorTone').textContent = (color.brightness || 'mid') + ' brightness, ' + (color.saturation || 'balanced');

        // Palette Swatches
        const pRow = document.getElementById('paletteRow');
        pRow.innerHTML = '';
        if (color.palette) {
          color.palette.forEach(p => {
            const swatch = document.createElement('div');
            swatch.className = 'swatch';
            swatch.style.backgroundColor = p.hex;
            swatch.title = p.hex + ' (' + Math.round(p.percent * 100) + '%)';
            pRow.appendChild(swatch);
          });
        }

        // Grad-CAM Heatmap
        const expl = data.explanation;
        if (expl && expl.overlay_url) {
          document.getElementById('camSection').style.display = 'block';
          document.getElementById('camOriginal').src = previewImg.src;
          document.getElementById('camOverlay').src = expl.overlay_url;
          document.getElementById('camCaption').textContent = (expl.target_prediction || explainTask) + ' Attribution';
        } else {
          document.getElementById('camSection').style.display = 'none';
        }

      } catch (err) {
        loadingSpinner.style.display = 'none';
        placeholder.style.display = 'block';
        alert('Analysis error: ' + err.message);
      } finally {
        analyzeBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
""")

# Mount built React frontend static assets if dist folder exists
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist"))
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="frontend-assets")


