import os
import math
import yaml
import colorsys
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans


class ColorPaletteExtractor:
    """
    K-Means based dominant color extraction in CIELAB color space,
    with near-duplicate cluster merging and rule-based temperature & tone classification.
    """
    def __init__(self, config_path: str = "configs/color.yaml"):
        if isinstance(config_path, str) and os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
        elif isinstance(config_path, dict):
            self.config = config_path
        else:
            # Fallback default configuration
            self.config = {
                "palette": {
                    "k_clusters": 5,
                    "max_side": 180,
                    "n_init": 5,
                    "seed": 42,
                    "merge_delta_e_threshold": 12.0
                },
                "temperature": {
                    "warm_hue_ranges": [[0, 65], [330, 360]],
                    "cool_hue_ranges": [[140, 270]],
                    "min_saturation_threshold": 0.15,
                    "min_chroma_threshold": 12.0,
                    "monochrome_max_saturation": 0.14,
                    "monochrome_hue_concentration": 0.85,
                    "dominant_ratio_threshold": 0.60,
                    "ratio_margin": 1.4
                },
                "tone": {
                    "brightness_thresholds": {"dark": 0.35, "light": 0.65},
                    "saturation_thresholds": {"muted": 0.25, "vivid": 0.55},
                    "pastel": {"min_lightness": 0.65, "min_saturation": 0.12, "max_saturation": 0.45}
                }
            }

        self.k_clusters = self.config.get("palette", {}).get("k_clusters", 5)
        self.max_side = self.config.get("palette", {}).get("max_side", 180)
        self.n_init = self.config.get("palette", {}).get("n_init", 5)
        self.seed = self.config.get("palette", {}).get("seed", 42)
        self.merge_delta_e = self.config.get("palette", {}).get("merge_delta_e_threshold", 12.0)

    def preprocess_image(self, image_input):
        """
        Loads and standardizes image:
        - Handles file path or PIL Image
        - Composites alpha transparency onto solid white
        - Detects pure grayscale
        - Downscales longest side to max_side preserving aspect ratio
        - Returns float32 RGB array normalized to [0, 1] and boolean is_grayscale flag
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Check pure grayscale
        is_grayscale = False
        if img.mode in ("L", "1", "I"):
            is_grayscale = True
        elif img.mode == "RGB":
            # Quick check if channels are identical
            arr = np.array(img)
            if np.all(arr[:, :, 0] == arr[:, :, 1]) and np.all(arr[:, :, 1] == arr[:, :, 2]):
                is_grayscale = True

        # Handle alpha channel (composite onto solid white)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            img = bg.convert("RGB")
        else:
            img = img.convert("RGB")

        # Downscale preserving aspect ratio
        w, h = img.size
        if max(w, h) > self.max_side:
            scale = self.max_side / max(w, h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        rgb_norm = np.array(img, dtype=np.float32) / 255.0
        return rgb_norm, is_grayscale

    def extract_palette(self, image_input, k: int = None):
        """
        Extracts dominant colors in CIELAB space, merges near-duplicates,
        and returns list of swatches with hex, rgb, and pixel percentage.
        """
        if k is None:
            k = self.k_clusters

        rgb_norm, is_grayscale = self.preprocess_image(image_input)
        h, w, _ = rgb_norm.shape
        pixels_rgb = rgb_norm.reshape(-1, 3)

        # Convert to CIELAB (OpenCV expects float32 in [0, 1] range)
        pixels_lab = cv2.cvtColor(rgb_norm, cv2.COLOR_RGB2LAB).reshape(-1, 3)

        # Check number of unique colors
        unique_colors = np.unique(pixels_lab, axis=0)
        n_unique = len(unique_colors)
        k_actual = max(1, min(k, n_unique))

        # K-Means clustering in CIELAB
        kmeans = KMeans(
            n_clusters=k_actual,
            random_state=self.seed,
            n_init=self.n_init
        )
        kmeans.fit(pixels_lab)

        centers_lab = kmeans.cluster_centers_
        labels = kmeans.labels_

        counts = np.bincount(labels, minlength=k_actual)
        weights = counts.astype(float) / float(len(labels))

        # Merge near-duplicate clusters (perceptual distance Delta-E < threshold in CIELAB)
        merged_centers = [list(c) for c in centers_lab]
        merged_weights = list(weights)

        if self.merge_delta_e > 0:
            while len(merged_centers) > 1:
                # Find closest pair
                min_dist = float("inf")
                min_pair = None
                for i in range(len(merged_centers)):
                    for j in range(i + 1, len(merged_centers)):
                        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(merged_centers[i], merged_centers[j])))
                        if dist < min_dist:
                            min_dist = dist
                            min_pair = (i, j)

                if min_dist < self.merge_delta_e and min_pair is not None:
                    i, j = min_pair
                    w_i, w_j = merged_weights[i], merged_weights[j]
                    w_tot = w_i + w_j
                    # Weighted centroid in LAB
                    new_center = [
                        (w_i * merged_centers[i][dim] + w_j * merged_centers[j][dim]) / w_tot
                        for dim in range(3)
                    ]
                    merged_centers[i] = new_center
                    merged_weights[i] = w_tot
                    del merged_centers[j]
                    del merged_weights[j]
                else:
                    break

        # Normalize weights to sum exactly to 1.0
        tot_w = sum(merged_weights)
        if tot_w > 0:
            merged_weights = [w / tot_w for w in merged_weights]

        # Convert centroids back from CIELAB to sRGB
        palette = []
        for lab_c, weight in zip(merged_centers, merged_weights):
            lab_arr = np.array([[lab_c]], dtype=np.float32)
            rgb_float = cv2.cvtColor(lab_arr, cv2.COLOR_LAB2RGB)[0, 0]
            
            r = int(round(np.clip(rgb_float[0], 0.0, 1.0) * 255))
            g = int(round(np.clip(rgb_float[1], 0.0, 1.0) * 255))
            b = int(round(np.clip(rgb_float[2], 0.0, 1.0) * 255))
            hex_code = f"#{r:02x}{g:02x}{b:02x}"

            # Hue, Saturation from HSV; Lightness from CIELAB L*
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            hue_deg = float(h * 360.0)
            lightness = float(np.clip(lab_c[0] / 100.0, 0.0, 1.0))
            saturation = float(np.clip(s, 0.0, 1.0))

            # CIELAB chroma
            chroma = math.sqrt(lab_c[1] ** 2 + lab_c[2] ** 2)

            palette.append({
                "hex": hex_code,
                "rgb": [r, g, b],
                "percent": round(float(weight), 4),
                "hue": round(hue_deg, 2),
                "lightness": round(lightness, 4),
                "saturation": round(saturation, 4),
                "chroma": round(float(chroma), 2)
            })

        # Sort descending by percent
        palette.sort(key=lambda x: x["percent"], reverse=True)
        return palette, is_grayscale

    def classify_temperature_and_tone(self, palette: list, is_grayscale: bool = False):
        """
        Rule-based classification of color temperature (warm, cool, neutral-mixed, monochrome)
        and tone descriptors (brightness, saturation, pastel flag).
        """
        temp_cfg = self.config.get("temperature", {})
        tone_cfg = self.config.get("tone", {})

        warm_ranges = temp_cfg.get("warm_hue_ranges", [[0, 65], [330, 360]])
        cool_ranges = temp_cfg.get("cool_hue_ranges", [[140, 270]])
        min_sat = temp_cfg.get("min_saturation_threshold", 0.15)
        mono_max_sat = temp_cfg.get("monochrome_max_saturation", 0.14)
        mono_narrow_max_sat = temp_cfg.get("monochrome_narrow_band_max_saturation", 0.28)
        mono_hue_conc = temp_cfg.get("monochrome_hue_concentration", 0.85)
        dom_thresh = temp_cfg.get("dominant_ratio_threshold", 0.60)
        ratio_margin = temp_cfg.get("ratio_margin", 1.4)

        if not palette:
            return {
                "temperature": "monochrome",
                "warm_score": 0.0,
                "cool_score": 0.0,
                "brightness": "mid",
                "saturation": "muted",
                "is_pastel": False
            }

        # Weighted mean saturation & lightness
        weighted_sat = sum(p["saturation"] * p["percent"] for p in palette)
        weighted_light = sum(p["lightness"] * p["percent"] for p in palette)

        # 1. Monochrome Check (Overall low saturation or sepia/duotone narrow band with low-mid saturation)
        is_narrow_band_monochrome = False
        chromatic_palette = [p for p in palette if p["saturation"] >= min_sat]
        tot_chromatic_mass = sum(p["percent"] for p in chromatic_palette)

        if tot_chromatic_mass > 0 and weighted_sat <= mono_narrow_max_sat:
            for base_angle in range(0, 360, 15):
                # Angular difference
                in_band = sum(
                    p["percent"] for p in chromatic_palette
                    if min(abs(p["hue"] - base_angle), 360 - abs(p["hue"] - base_angle)) <= 12.5
                )
                if in_band / tot_chromatic_mass >= mono_hue_conc:
                    is_narrow_band_monochrome = True
                    break

        if is_grayscale or weighted_sat < mono_max_sat or is_narrow_band_monochrome:
            temperature = "monochrome"
            warm_score = 0.0
            cool_score = 0.0
        else:
            # 2. Warm / Cool Voting
                warm_vote = 0.0
                cool_vote = 0.0
                total_chroma_vote = 0.0

                def in_ranges(val, ranges):
                    for r_min, r_max in ranges:
                        if r_min <= val <= r_max:
                            return True
                    return False

                for p in palette:
                    sat = p["saturation"]
                    pct = p["percent"]
                    hue = p["hue"]

                    if sat >= min_sat:
                        vote = pct * sat
                        total_chroma_vote += vote

                        if in_ranges(hue, warm_ranges):
                            warm_vote += vote
                        elif in_ranges(hue, cool_ranges):
                            cool_vote += vote

                if total_chroma_vote > 0:
                    raw_warm_ratio = warm_vote / total_chroma_vote
                    raw_cool_ratio = cool_vote / total_chroma_vote
                    warm_score = round(float(raw_warm_ratio), 4)
                    cool_score = round(float(raw_cool_ratio), 4)
                else:
                    warm_score = 0.5
                    cool_score = 0.5

                # Decision rule
                if warm_score >= dom_thresh and (cool_score == 0 or warm_score >= ratio_margin * cool_score):
                    temperature = "warm"
                elif cool_score >= dom_thresh and (warm_score == 0 or cool_score >= ratio_margin * warm_score):
                    temperature = "cool"
                else:
                    temperature = "neutral-mixed"

        # 3. Tone Descriptors
        b_thresh = tone_cfg.get("brightness_thresholds", {"dark": 0.35, "light": 0.65})
        s_thresh = tone_cfg.get("saturation_thresholds", {"muted": 0.25, "vivid": 0.55})
        pastel_cfg = tone_cfg.get("pastel", {"min_lightness": 0.65, "min_saturation": 0.12, "max_saturation": 0.45})

        if weighted_light < b_thresh.get("dark", 0.35):
            brightness = "dark"
        elif weighted_light > b_thresh.get("light", 0.65):
            brightness = "light"
        else:
            brightness = "mid"

        if weighted_sat < s_thresh.get("muted", 0.25):
            saturation = "muted"
        elif weighted_sat > s_thresh.get("vivid", 0.55):
            saturation = "vivid"
        else:
            saturation = "balanced"

        is_pastel = bool(
            weighted_light >= pastel_cfg.get("min_lightness", 0.65)
            and pastel_cfg.get("min_saturation", 0.12) <= weighted_sat <= pastel_cfg.get("max_saturation", 0.45)
        )

        return {
            "temperature": temperature,
            "warm_score": warm_score,
            "cool_score": cool_score,
            "brightness": brightness,
            "saturation": saturation,
            "is_pastel": is_pastel
        }

    def analyze_image(self, image_input, k: int = None):
        """
        End-to-end extraction and classification.
        Returns a clean dictionary matching the required JSON schema.
        """
        palette, is_grayscale = self.extract_palette(image_input, k=k)
        tone_info = self.classify_temperature_and_tone(palette, is_grayscale=is_grayscale)

        # Simplify palette output to clean public schema (hex, rgb, percent)
        clean_palette = [
            {"hex": p["hex"], "rgb": p["rgb"], "percent": p["percent"]}
            for p in palette
        ]

        return {
            "palette": clean_palette,
            "color_count": len(clean_palette),
            "temperature": tone_info["temperature"],
            "warm_score": tone_info["warm_score"],
            "cool_score": tone_info["cool_score"],
            "brightness": tone_info["brightness"],
            "saturation": tone_info["saturation"],
            "is_pastel": tone_info["is_pastel"]
        }
