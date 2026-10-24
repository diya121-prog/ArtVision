import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from predict import predict_artwork


def test_predict_schema():
    image_path = "data/processed/images/ai_artbench_0-114513684-761761.jpg"
    checkpoint_path = "runs/multitask_dev/best_model.pth"

    res = predict_artwork(image_path, checkpoint_path)
    
    # Assert top-level keys
    required_keys = ["style", "movement", "artist", "genre", "mood", "authenticity", "era", "technique", "color"]
    for k in required_keys:
        assert k in res, f"Missing key '{k}' in predict output"

    # Style
    assert "top3" in res["style"] and len(res["style"]["top3"]) == 3
    assert "confidence" in res["style"]
    assert "uncertain" in res["style"] and isinstance(res["style"]["uncertain"], bool)
    
    # Movement
    assert "derived_from_style" in res["movement"]
    assert "movement" in res["movement"]
    
    # Artist
    assert "top3" in res["artist"]
    assert "uncertain" in res["artist"] and isinstance(res["artist"]["uncertain"], bool)
    
    # Genre, Era, Technique
    assert res["genre"] is None
    assert res["era"] is None
    assert res["technique"] is None
    
    # Mood
    assert "distribution" in res["mood"]
    assert len(res["mood"]["distribution"]) == 5
    assert "uncertain" in res["mood"] and isinstance(res["mood"]["uncertain"], bool)
    
    # Authenticity
    assert "probability_ai" in res["authenticity"]
    assert "is_ai_generated" in res["authenticity"]
    assert "label" in res["authenticity"]
    assert "uncertain" in res["authenticity"] and isinstance(res["authenticity"]["uncertain"], bool)

    # Color (Phase 5)
    assert res["color"] is not None
    assert "palette" in res["color"]
    assert "temperature" in res["color"]
    assert res["color"]["temperature"] in ["warm", "cool", "neutral-mixed", "monochrome"]
    assert "warm_score" in res["color"] and 0.0 <= res["color"]["warm_score"] <= 1.0
    assert "cool_score" in res["color"] and 0.0 <= res["color"]["cool_score"] <= 1.0
    assert "brightness" in res["color"] and res["color"]["brightness"] in ["dark", "mid", "light"]
    assert "saturation" in res["color"] and res["color"]["saturation"] in ["muted", "balanced", "vivid"]
    assert "is_pastel" in res["color"] and isinstance(res["color"]["is_pastel"], bool)
    assert len(res["color"]["palette"]) > 0
    assert "hex" in res["color"]["palette"][0]
    assert "rgb" in res["color"]["palette"][0]
    assert "percent" in res["color"]["palette"][0]

    print("predict.py schema test passed!")


if __name__ == "__main__":
    test_predict_schema()
