import os
import json
import unittest
import numpy as np

from src.features.search import ArtworkSearcher


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        emb_path = "runs/retrieval/embeddings.npy"
        ids_path = "runs/retrieval/image_ids.json"
        if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
            raise unittest.SkipTest("Retrieval embeddings not yet built")
        cls.searcher = ArtworkSearcher(
            embeddings_path=emb_path,
            image_ids_path=ids_path,
            checkpoint_path="runs/multitask_model/best_model.pth"
        )

    def test_embeddings_matrix_properties(self):
        embs = self.searcher.embeddings
        self.assertEqual(embs.ndim, 2)
        self.assertEqual(embs.shape[1], 1280)
        self.assertEqual(embs.shape[0], 8099)

        # Verify L2 normalization
        norms = np.linalg.norm(embs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_self_exclusion_and_near_duplicate_filter(self):
        test_id = self.searcher.metadata[0]["image_id"]
        results = self.searcher.find_similar(test_id, top_k=5, max_sim=0.995)

        self.assertGreater(len(results), 0)
        retrieved_ids = [r["image_id"] for r in results]

        # Verify self-exclusion
        self.assertNotIn(test_id, retrieved_ids, f"Self '{test_id}' was returned in search results!")

        # Verify near-duplicate filter
        for r in results:
            self.assertLessEqual(r["similarity"], 0.995)
            self.assertTrue(-1.0 <= r["similarity"] <= 1.0)
            self.assertIn("style", r)
            self.assertIn("artist", r)
            self.assertIn("genre", r)


if __name__ == "__main__":
    unittest.main()
