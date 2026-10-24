import os
import sys
import re
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.features.color_palette import ColorPaletteExtractor


def test_synthetic_images():
    extractor = ColorPaletteExtractor("configs/color.yaml")

    # 1. Solid Pure Red
    red_img = Image.new("RGB", (100, 100), (255, 0, 0))
    res_red = extractor.analyze_image(red_img)
    assert res_red["temperature"] == "warm", f"Expected warm, got {res_red['temperature']}"
    assert res_red["warm_score"] == 1.0
    assert res_red["color_count"] == 1
    assert res_red["palette"][0]["hex"] == "#ff0000"
    assert abs(res_red["palette"][0]["percent"] - 1.0) < 1e-3

    # 2. Solid Pure Blue
    blue_img = Image.new("RGB", (100, 100), (0, 0, 255))
    res_blue = extractor.analyze_image(blue_img)
    assert res_blue["temperature"] == "cool", f"Expected cool, got {res_blue['temperature']}"
    assert res_blue["cool_score"] == 1.0
    assert res_blue["palette"][0]["hex"] == "#0000ff"

    # 3. Solid Pure Gray
    gray_img = Image.new("RGB", (100, 100), (128, 128, 128))
    res_gray = extractor.analyze_image(gray_img)
    assert res_gray["temperature"] == "monochrome", f"Expected monochrome, got {res_gray['temperature']}"
    assert res_gray["warm_score"] == 0.0
    assert res_gray["cool_score"] == 0.0

    # 4. Red-Blue Split 70% Red / 30% Blue
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:, :70] = [255, 0, 0]
    arr[:, 70:] = [0, 0, 255]
    split_img = Image.fromarray(arr)
    res_split = extractor.analyze_image(split_img)
    assert res_split["temperature"] == "warm", f"Expected warm, got {res_split['temperature']}"
    assert res_split["warm_score"] >= 0.65, f"Expected warm_score >= 0.65, got {res_split['warm_score']}"
    assert res_split["cool_score"] >= 0.20, f"Expected cool_score >= 0.20, got {res_split['cool_score']}"

    # 5. Pastel Pink Image (Lightness ~0.85, Saturation ~0.25)
    pastel_img = Image.new("RGB", (100, 100), (245, 195, 205))
    res_pastel = extractor.analyze_image(pastel_img)
    assert res_pastel["is_pastel"] is True, f"Expected pastel=True, got {res_pastel}"
    assert res_pastel["brightness"] == "light"

    print("Synthetic image tests passed!")


def test_edge_cases():
    extractor = ColorPaletteExtractor("configs/color.yaml")

    # 1. 1-Channel Grayscale (Mode 'L')
    gray_l = Image.new("L", (80, 80), 100)
    res_l = extractor.analyze_image(gray_l)
    assert res_l["temperature"] == "monochrome"

    # 2. RGBA with Alpha Transparency
    rgba_arr = np.zeros((60, 60, 4), dtype=np.uint8)
    rgba_arr[:, :30] = [255, 0, 0, 255]       # Opaque red
    rgba_arr[:, 30:] = [0, 0, 0, 0]           # Completely transparent (should composite to white)
    rgba_img = Image.fromarray(rgba_arr, mode="RGBA")
    res_rgba = extractor.analyze_image(rgba_img)
    assert res_rgba["color_count"] >= 1
    # Check that white background was composited
    hex_list = [p["hex"] for p in res_rgba["palette"]]
    assert "#ffffff" in hex_list

    # 3. Extremely Small Images (3x3 and 1x1)
    tiny_img = Image.new("RGB", (3, 3), (200, 50, 50))
    res_tiny = extractor.analyze_image(tiny_img)
    assert res_tiny["color_count"] == 1
    assert res_tiny["palette"][0]["percent"] == 1.0

    one_px = Image.new("RGB", (1, 1), (0, 150, 200))
    res_one = extractor.analyze_image(one_px)
    assert res_one["color_count"] == 1

    # 4. Image with 2 Colors when k=5
    two_col = np.zeros((50, 50, 3), dtype=np.uint8)
    two_col[:, :25] = [255, 200, 0]
    two_col[:, 25:] = [0, 100, 200]
    res_two = extractor.analyze_image(Image.fromarray(two_col))
    assert res_two["color_count"] <= 2

    print("Edge case tests passed!")


def test_determinism_and_formats():
    extractor = ColorPaletteExtractor("configs/color.yaml")
    test_img = Image.new("RGB", (120, 120))
    arr = np.array(test_img)
    # Checkerboard with several colors
    arr[:60, :60] = [180, 40, 40]
    arr[:60, 60:] = [40, 180, 40]
    arr[60:, :60] = [40, 40, 180]
    arr[60:, 60:] = [220, 220, 80]
    img = Image.fromarray(arr)

    res1 = extractor.analyze_image(img)
    res2 = extractor.analyze_image(img)

    # Determinism
    assert res1 == res2, "Extraction must be strictly deterministic!"

    # Format & Percentages
    hex_pattern = re.compile(r"^#[0-9a-fA-F]{6}$")
    total_pct = sum(p["percent"] for p in res1["palette"])
    assert abs(total_pct - 1.0) < 0.01, f"Percentages should sum to ~1.0, got {total_pct}"

    for p in res1["palette"]:
        assert hex_pattern.match(p["hex"]), f"Invalid hex code: {p['hex']}"
        assert len(p["rgb"]) == 3
        for val in p["rgb"]:
            assert 0 <= val <= 255

    print("Determinism and format tests passed!")


def test_near_duplicate_merging():
    extractor = ColorPaletteExtractor("configs/color.yaml")
    # Two shades of green that are nearly identical (Delta E < 12.0 in CIELAB)
    arr = np.zeros((80, 80, 3), dtype=np.uint8)
    arr[:, :40] = [20, 150, 30]
    arr[:, 40:] = [22, 153, 32]
    img = Image.fromarray(arr)

    res = extractor.analyze_image(img, k=5)
    # The two near-duplicate green clusters should be merged into a single cluster
    assert res["color_count"] == 1, f"Expected 1 merged cluster, got {res['color_count']}"
    assert abs(res["palette"][0]["percent"] - 1.0) < 1e-3

    print("Near-duplicate merging test passed!")


if __name__ == "__main__":
    test_synthetic_images()
    test_edge_cases()
    test_determinism_and_formats()
    test_near_duplicate_merging()
    print("\nALL COLOR PALETTE UNIT TESTS PASSED!")
