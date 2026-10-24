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


# Mount built React frontend SPA if dist folder exists
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend/dist"))
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

