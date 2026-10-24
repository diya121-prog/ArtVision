import os
import sys
import unittest
import tempfile
import shutil
import hashlib
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import imagehash

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.build_wikiart_cache import process_and_save_image, load_existing_cached_image


class TestCachingResume(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.img_dir = os.path.join(self.test_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.staging_path = os.path.join(self.test_dir, "staging_test.csv")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_phash_sanity_check(self):
        """Sanity check pHash distance on known identical, near-duplicate, and distinct images."""
        real_img_path = "data/processed/images/ai_artbench_0-114513684-761761.jpg"
        if os.path.exists(real_img_path):
            base_img = Image.open(real_img_path).convert("RGB")
        else:
            base_img = Image.new("RGB", (200, 200), color=(120, 80, 200))
            for x in range(0, 200, 10):
                for y in range(0, 200, 10):
                    base_img.putpixel((x, y), ((x * 3) % 255, (y * 5) % 255, (x * y) % 255))

        # 1. Exact copy
        h_base = imagehash.phash(base_img)
        h_exact = imagehash.phash(base_img)
        self.assertEqual(h_base - h_exact, 0, "Exact duplicate must have pHash distance 0")

        # 2. Near-duplicate (slight brightness shift)
        enhancer = ImageEnhance.Brightness(base_img)
        near_img = enhancer.enhance(1.05)
        h_near = imagehash.phash(near_img)
        self.assertLessEqual(h_base - h_near, 4, "Near duplicate must have pHash distance <= 4")

        # 3. Distinct image
        diff_img_path = "data/processed/images/ai_artbench_0-126543674-86853.jpg"
        if os.path.exists(diff_img_path):
            diff_img = Image.open(diff_img_path).convert("RGB")
        else:
            diff_img = Image.new("RGB", (200, 200), color=(10, 220, 30))
            for x in range(0, 200, 5):
                diff_img.putpixel((x, x), (0, 0, 0))
        h_diff = imagehash.phash(diff_img)
        self.assertGreater(h_base - h_diff, 4, "Distinct image must have pHash distance > 4")

    def test_interruption_and_resume_simulation(self):
        """Simulate an interrupted ingestion run and verify resume idempotency."""
        staged_rows = []
        file_mtimes = {}

        for i in range(1, 6):
            img_id = f"test_{i:04d}"
            save_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            rng = np.random.RandomState(i * 100)
            arr = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            md5, phash_str, phash_bool, w, h = process_and_save_image(img, 384, 90, save_path)
            
            staged_rows.append({
                "image_id": img_id,
                "filepath": save_path,
                "source_dataset": "mock_wikiart",
                "style": f"Style_{i}",
                "movement": "Modern",
                "artist": f"Artist_{i}",
                "genre": "Landscape",
                "file_hash": md5,
                "phash": str(phash_str),
                "width": w,
                "height": h,
                "split": ""
            })
            file_mtimes[save_path] = os.path.getmtime(save_path)

        pd.DataFrame(staged_rows).to_csv(self.staging_path, index=False)
        self.assertTrue(os.path.exists(self.staging_path))
        self.assertEqual(len(pd.read_csv(self.staging_path)), 5)

        # Resume simulation
        resumed_df = pd.read_csv(self.staging_path, dtype={"phash": str})
        resumed_rows = resumed_df.to_dict(orient="records")
        resumed_md5s = set(resumed_df["file_hash"].dropna().values)
        resumed_phashes = [imagehash.hex_to_hash(str(h)).hash.flatten() for h in resumed_df["phash"].dropna().values]
        phash_matrix = np.array(resumed_phashes, dtype=bool)

        self.assertEqual(len(resumed_rows), 5)
        collected_count = len(resumed_rows)

        # Process items 6 to 10
        for i in range(6, 11):
            img_id = f"test_{i:04d}"
            save_path = os.path.join(self.img_dir, f"{img_id}.jpg")
            rng = np.random.RandomState(i * 100)
            arr = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            
            if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                md5, phash_str, phash_bool, w, h = load_existing_cached_image(save_path)
            else:
                md5, phash_str, phash_bool, w, h = process_and_save_image(img, 384, 90, save_path)

            self.assertNotIn(md5, resumed_md5s)
            dists = np.count_nonzero(phash_matrix != phash_bool, axis=1)
            self.assertFalse(np.any(dists <= 4))

            resumed_md5s.add(md5)
            phash_matrix = np.vstack([phash_matrix, phash_bool])

            resumed_rows.append({
                "image_id": img_id,
                "filepath": save_path,
                "source_dataset": "mock_wikiart",
                "style": f"Style_{i}",
                "movement": "Modern",
                "artist": f"Artist_{i}",
                "genre": "Landscape",
                "file_hash": md5,
                "phash": phash_str,
                "width": w,
                "height": h,
                "split": ""
            })
            collected_count += 1

        pd.DataFrame(resumed_rows).to_csv(self.staging_path, index=False)

        final_df = pd.read_csv(self.staging_path)
        self.assertEqual(len(final_df), 10, "Final staging must have exactly 10 rows")
        self.assertEqual(final_df["file_hash"].nunique(), 10, "All 10 file hashes must be unique")

        for i in range(1, 6):
            save_path = os.path.join(self.img_dir, f"test_{i:04d}.jpg")
            self.assertEqual(os.path.getmtime(save_path), file_mtimes[save_path], f"File {save_path} was overwritten!")


if __name__ == "__main__":
    unittest.main()
