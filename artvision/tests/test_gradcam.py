import os
import unittest
import numpy as np
import torch
from PIL import Image

from src.models.multitask import MultiTaskModel
from src.explain.gradcam import GradCAM, overlay_heatmap, calculate_border_mass_fraction


class TestGradCAM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ckpt_path = "runs/multitask_model/best_model.pth"
        if not os.path.exists(ckpt_path):
            raise unittest.SkipTest("Model checkpoint not found")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        cls.model = MultiTaskModel(ckpt["config"])
        cls.model.load_state_dict(ckpt["model_state_dict"])
        cls.model.eval()
        cls.dummy_input = torch.randn(1, 3, 224, 224)

    def test_gradcam_hook_lifecycle(self):
        gc = GradCAM(self.model)
        self.assertEqual(len(gc.hooks), 0, "Hooks should be empty initially")
        with gc:
            self.assertEqual(len(gc.hooks), 2, "Should register forward and backward hooks inside context")
            cam = gc.generate_heatmap(self.dummy_input, task="style")
            self.assertIsNotNone(gc.activations)
            self.assertIsNotNone(gc.gradients)

        self.assertEqual(len(gc.hooks), 0, "Hooks must be cleanly removed upon exiting context")
        self.assertIsNone(gc.activations, "Activations reference must be cleared")
        self.assertIsNone(gc.gradients, "Gradients reference must be cleared")

    def test_gradcam_output_shape_and_bounds(self):
        with GradCAM(self.model) as gc:
            for task in ["style", "artist", "authenticity"]:
                cam = gc.generate_heatmap(self.dummy_input, task=task)
                self.assertIsInstance(cam, np.ndarray)
                self.assertEqual(cam.shape, (224, 224), f"Heatmap shape should be (224, 224), got {cam.shape}")
                self.assertTrue(np.all(cam >= 0.0), f"All values must be >= 0.0, min was {cam.min()}")
                self.assertTrue(np.all(cam <= 1.0), f"All values must be <= 1.0, max was {cam.max()}")

    def test_gradcam_overlay_and_border_mass(self):
        cam = np.random.uniform(0.0, 1.0, size=(224, 224))
        pil_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
        overlay = overlay_heatmap(cam, pil_img, alpha=0.5)
        self.assertIsInstance(overlay, Image.Image)
        self.assertEqual(overlay.size, (256, 256))

        bmf = calculate_border_mass_fraction(cam, border_ratio=0.1)
        self.assertIsInstance(bmf, float)
        self.assertTrue(0.0 <= bmf <= 1.0)


if __name__ == "__main__":
    unittest.main()
