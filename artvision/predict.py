import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import json
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.models.multitask import MultiTaskModel
from src.training.calibration import TemperatureScaler
from src.features.color_palette import ColorPaletteExtractor
from src.explain.gradcam import GradCAM, overlay_heatmap, calculate_border_mass_fraction

_cached_single_auth_model = None
_cached_single_auth_path = None

def get_single_auth_model(ckpt_path: str, device):
    global _cached_single_auth_model, _cached_single_auth_path
    if _cached_single_auth_model is not None and _cached_single_auth_path == ckpt_path:
        return _cached_single_auth_model
    if os.path.exists(ckpt_path):
        try:
            s_ckpt = torch.load(ckpt_path, map_location="cpu")
            m_auth = MultiTaskModel(s_ckpt["config"])
            m_auth.load_state_dict(s_ckpt["model_state_dict"])
            m_auth.to(device)
            m_auth.eval()
            _cached_single_auth_model = m_auth
            _cached_single_auth_path = ckpt_path
            return _cached_single_auth_model
        except Exception:
            return None
    return None

def predict_artwork(
    image_path: str,
    checkpoint_path: str,
    calibration_path: str = None,
    color_config_path: str = None,
    predict_config_path: str = None,
    device: str = None,
    explain: bool = False,
    explain_task: str = "style",
    explain_output_dir: str = "runs/explain"
):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt["config"]

    # Load predict / uncertainty config
    if predict_config_path is None:
        default_predict_cfg = os.path.join(os.path.dirname(__file__), "configs/predict.yaml")
        if os.path.exists(default_predict_cfg):
            predict_config_path = default_predict_cfg

    uncertainty_cfg = {}
    if predict_config_path and os.path.exists(predict_config_path):
        with open(predict_config_path) as f:
            pcfg = yaml.safe_load(f)
            uncertainty_cfg = pcfg.get("uncertainty", {})

    def is_uncertain(probs, task_name):
        t_cfg = uncertainty_cfg.get(task_name, {})
        min_conf = t_cfg.get("min_confidence", 0.0)
        min_margin = t_cfg.get("min_margin", 0.0)
        sorted_p = sorted(probs)
        c1 = sorted_p[-1] if len(sorted_p) > 0 else 0.0
        c2 = sorted_p[-2] if len(sorted_p) > 1 else 0.0
        margin = c1 - c2
        return bool(c1 < min_conf or margin < min_margin)

    def is_uncertain_binary(prob, task_name):
        t_cfg = uncertainty_cfg.get(task_name, {})
        min_conf = t_cfg.get("min_confidence", 0.0)
        min_margin = t_cfg.get("min_margin", 0.0)
        conf = max(prob, 1.0 - prob)
        margin = abs(prob - (1.0 - prob))
        return bool(conf < min_conf or margin < min_margin)

    # Load label maps
    label_maps_path = os.path.join(os.path.dirname(__file__), "configs/label_maps.yaml")
    with open(label_maps_path) as f:
        label_maps = yaml.safe_load(f)
    movement_map = label_maps.get("movement_map", {})
    era_map = label_maps.get("era_map", {})
    technique_map = label_maps.get("technique_map", {})

    style_classes = ckpt.get("style_classes", [])
    artist_classes = ckpt.get("artist_classes", [])
    mood_classes = ckpt.get("mood_classes", [])
    genre_classes = ckpt.get("genre_classes", [])

    # Load Model
    model = MultiTaskModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load Calibration
    scaler = TemperatureScaler()
    if calibration_path is None:
        auto_calib = os.path.join(os.path.dirname(checkpoint_path), "calibration.json")
        if os.path.exists(auto_calib):
            calibration_path = auto_calib

    if calibration_path and os.path.exists(calibration_path):
        scaler.load(calibration_path)

    # Transform
    img_size = config.get("data", {}).get("image_size", 224)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

    result = {}

    # 1. Style & Derived Movement
    if "style" in outputs and len(style_classes) > 0:
        logits = outputs["style"]
        T = scaler.temperatures.get("style", 1.0)
        probs = F.softmax(logits / T, dim=-1).squeeze(0).cpu().numpy()

        top_k = min(3, len(style_classes))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_styles = [
            {"style": style_classes[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
        top1_style = top_styles[0]["style"]
        top1_conf = top_styles[0]["confidence"]
        uncertain_flag = is_uncertain(probs, "style")
        result["style"] = {
            "top3": top_styles,
            "prediction": top1_style,
            "confidence": top1_conf,
            "uncertain": uncertain_flag
        }

        # Derived movement via style mapping
        result["movement"] = {
            "derived_from_style": top1_style,
            "movement": movement_map.get(top1_style, "Unknown Movement")
        }
    else:
        result["style"] = None
        result["movement"] = None

    # 2. Artist
    if "artist" in outputs and len(artist_classes) > 0:
        logits = outputs["artist"]
        T = scaler.temperatures.get("artist", 1.0)
        probs = F.softmax(logits / T, dim=-1).squeeze(0).cpu().numpy()

        top_k = min(3, len(artist_classes))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_artists = [
            {"artist": artist_classes[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
        top1_conf = top_artists[0]["confidence"]
        uncertain_flag = is_uncertain(probs, "artist")
        result["artist"] = {
            "top3": top_artists,
            "prediction": top_artists[0]["artist"],
            "confidence": top1_conf,
            "uncertain": uncertain_flag
        }
    else:
        result["artist"] = None

    # 3. Genre (Return null if no real labels)
    if "genre" in outputs and len(genre_classes) > 0 and genre_classes[0] != "genre_0":
        logits = outputs["genre"]
        T = scaler.temperatures.get("genre", 1.0)
        probs = F.softmax(logits / T, dim=-1).squeeze(0).cpu().numpy()
        top_k = min(3, len(genre_classes))
        top_indices = np.argsort(probs)[-top_k:][::-1]
        top_genres = [
            {"genre": genre_classes[idx], "confidence": round(float(probs[idx]), 4)}
            for idx in top_indices
        ]
        uncertain_flag = is_uncertain(probs, "genre")
        result["genre"] = {
            "top3": top_genres,
            "prediction": top_genres[0]["genre"],
            "confidence": top_genres[0]["confidence"],
            "uncertain": uncertain_flag
        }
    else:
        # Explicitly null per specification
        result["genre"] = None

    # 4. Mood
    if "mood" in outputs and len(mood_classes) > 0:
        logits = outputs["mood"]
        T = scaler.temperatures.get("mood", 1.0)
        probs = F.softmax(logits / T, dim=-1).squeeze(0).cpu().numpy()
        mood_dist = {
            mood_classes[i]: round(float(probs[i]), 4)
            for i in range(len(mood_classes))
        }
        top_mood_idx = int(np.argmax(probs))
        uncertain_flag = is_uncertain(probs, "mood")
        result["mood"] = {
            "prediction": mood_classes[top_mood_idx],
            "confidence": round(float(probs[top_mood_idx]), 4),
            "distribution": mood_dist,
            "uncertain": uncertain_flag,
            "experimental": True
        }
    else:
        result["mood"] = None

    # 5. Authenticity (Binary is_ai_generated)
    if "authenticity" in outputs:
        logit = outputs["authenticity"]
        T = scaler.temperatures.get("authenticity", 1.0)
        prob_ai_multi = float(torch.sigmoid(logit / T).squeeze().cpu().item())

        # Check dedicated single-task authenticity model if available for ensemble
        single_auth_ckpt = os.path.join(os.path.dirname(checkpoint_path), "../single_task_auth/best_model.pth")
        single_auth_model = get_single_auth_model(single_auth_ckpt, device)
        if single_auth_model is not None:
            try:
                with torch.no_grad():
                    s_out = single_auth_model(tensor)
                    prob_ai_single = float(torch.sigmoid(s_out["authenticity"]).squeeze().cpu().item())
                    # Weighted ensemble prioritizing high confidence detection
                    raw_prob_ai = max(prob_ai_multi, 0.4 * prob_ai_multi + 0.6 * prob_ai_single)
            except Exception:
                raw_prob_ai = prob_ai_multi
        else:
            raw_prob_ai = prob_ai_multi

        # Bayes-calibrated threshold to compensate for training prior imbalance (87% human, 13% AI)
        is_ai = raw_prob_ai >= 0.38
        if is_ai:
            calibrated_prob = 0.50 + 0.50 * ((raw_prob_ai - 0.38) / (1.0 - 0.38))
            calibrated_prob = min(0.999, max(0.52, calibrated_prob))
        else:
            calibrated_prob = 0.50 * (raw_prob_ai / 0.38)
            calibrated_prob = min(0.48, max(0.001, calibrated_prob))

        conf = calibrated_prob if is_ai else (1.0 - calibrated_prob)
        uncertain_flag = is_uncertain_binary(raw_prob_ai, "authenticity")
        result["authenticity"] = {
            "is_ai_generated": is_ai,
            "label": "ai_generated" if is_ai else "human_made",
            "probability_ai": round(calibrated_prob, 4),
            "confidence": round(conf, 4),
            "uncertain": uncertain_flag,
            "caveat": "Trained on specific generators; less reliable on resized, compressed or screenshotted images."
        }
    else:
        result["authenticity"] = None

    # 6. Era & Technique (Derived via Style, Artist & Authenticity analysis)
    top1_style = result.get("style", {}).get("prediction") if result.get("style") else None
    is_ai_detected = result.get("authenticity", {}).get("is_ai_generated", False) if result.get("authenticity") else False

    if is_ai_detected:
        result["era"] = {
            "period": "Contemporary Digital / Synthetic Synthesis",
            "timeline": "2020s Generative AI Era",
            "derived_from": "AI Authenticity Head"
        }
        result["technique"] = {
            "medium": "Latent Diffusion / Deep Neural Synthesis",
            "method": "Iterative denoising in latent feature space (text-to-image synthesis)",
            "derived_from": "AI Authenticity Head"
        }
    elif top1_style:
        era_info = era_map.get(top1_style, {
            "period": "Historical Artwork Era",
            "timeline": "Derived from style classification"
        })
        tech_info = technique_map.get(top1_style, {
            "medium": "Oil / tempera on canvas or wood panel",
            "method": "Traditional classical artistic technique"
        })
        result["era"] = {
            "period": era_info.get("period", "Historical Artwork Era"),
            "timeline": era_info.get("timeline", "Historical timeline"),
            "derived_from": top1_style
        }
        result["technique"] = {
            "medium": tech_info.get("medium", "Oil on canvas"),
            "method": tech_info.get("method", "Classical painting technique"),
            "derived_from": top1_style
        }
    else:
        result["era"] = None
        result["technique"] = None

    # 7. Color Palette & Tone Analysis (Phase 5)
    try:
        if color_config_path is None:
            default_color_cfg = os.path.join(os.path.dirname(__file__), "configs/color.yaml")
            if os.path.exists(default_color_cfg):
                color_config_path = default_color_cfg
        color_extractor = ColorPaletteExtractor(color_config_path)
        color_res = color_extractor.analyze_image(image_path)
        result["color"] = color_res
    except Exception as e:
        result["color"] = None
        result["color_error"] = str(e)



    # 8. Grad-CAM Visual Attribution
    if explain:
        try:
            os.makedirs(explain_output_dir, exist_ok=True)
            with GradCAM(model) as gc:
                # Determine target class index
                target_cls = None
                target_name = "top_prediction"
                if explain_task == "style" and result.get("style"):
                    target_name = result["style"]["prediction"]
                    if target_name in style_classes:
                        target_cls = style_classes.index(target_name)
                elif explain_task == "artist" and result.get("artist"):
                    target_name = result["artist"]["prediction"]
                    if target_name in artist_classes:
                        target_cls = artist_classes.index(target_name)
                elif explain_task == "genre" and result.get("genre"):
                    target_name = result["genre"]["prediction"]
                    if target_name in genre_classes:
                        target_cls = genre_classes.index(target_name)
                elif explain_task == "mood" and result.get("mood"):
                    target_name = result["mood"]["prediction"]
                    if target_name in mood_classes:
                        target_cls = mood_classes.index(target_name)
                elif explain_task == "authenticity" and result.get("authenticity"):
                    target_name = result["authenticity"]["label"]
                    target_cls = 1 if result["authenticity"]["is_ai_generated"] else 0

                cam = gc.generate_heatmap(tensor.clone(), task=explain_task, target_class=target_cls)
                bmf = calculate_border_mass_fraction(cam, border_ratio=0.1)

                overlay = overlay_heatmap(cam, image)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                overlay_filename = f"{base_name}_{explain_task}_gradcam.jpg"
                overlay_filepath = os.path.join(explain_output_dir, overlay_filename)
                overlay.save(overlay_filepath, "JPEG", quality=90)

                result["explanation"] = {
                    "task": explain_task,
                    "target_prediction": target_name,
                    "overlay_path": overlay_filepath,
                    "border_mass_fraction": round(float(bmf), 4),
                    "heatmap_dimensions": list(cam.shape)
                }
        except Exception as e:
            result["explanation"] = {
                "error": str(e)
            }
    else:
        result["explanation"] = None

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task artwork prediction CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input artwork image")
    parser.add_argument("--checkpoint", type=str, default="runs/multitask_model/best_model.pth", help="Model checkpoint")
    parser.add_argument("--calibration", type=str, default=None, help="Calibration JSON file")
    parser.add_argument("--color-config", type=str, default=None, help="Color palette config YAML")
    parser.add_argument("--predict-config", type=str, default=None, help="Predict / uncertainty config YAML")
    parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM visual explanation overlay")
    parser.add_argument("--explain-task", type=str, default="style", choices=["style", "artist", "genre", "mood", "authenticity"], help="Task head to explain")
    parser.add_argument("--explain-output-dir", type=str, default="runs/explain", help="Directory to save Grad-CAM overlay")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON filepath")
    args = parser.parse_args()

    pred = predict_artwork(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        calibration_path=args.calibration,
        color_config_path=args.color_config,
        predict_config_path=args.predict_config,
        explain=args.explain,
        explain_task=args.explain_task,
        explain_output_dir=args.explain_output_dir
    )
    json_output = json.dumps(pred, indent=2)
    print(json_output)

    if args.output:
        with open(args.output, "w") as f:
            f.write(json_output)



