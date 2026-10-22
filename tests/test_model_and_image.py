import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    from predict import prepare_digit_image
    from train import MultiInputCNN, auxiliary_features
except ModuleNotFoundError as error:
    if error.name not in {"numpy", "torch", "PIL", "sklearn", "matplotlib"}:
        raise
    np = torch = Image = ImageDraw = None
    prepare_digit_image = MultiInputCNN = auxiliary_features = None


@unittest.skipUnless(
    MultiInputCNN is not None,
    "Install multi-input-ocr-classifier requirements.",
)
class ModelAndImageTests(unittest.TestCase):
    def test_auxiliary_features_are_finite_and_four_dimensional(self):
        images = np.zeros((3, 8, 8), dtype=np.float32)
        images[:, 2:6, 3:5] = 1.0
        features = auxiliary_features(images)
        self.assertEqual(features.shape, (3, 4))
        self.assertTrue(np.isfinite(features).all())

    def test_model_fuses_image_and_auxiliary_inputs(self):
        model = MultiInputCNN()
        logits = model(torch.zeros(2, 1, 8, 8), torch.zeros(2, 4))
        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertTrue(torch.isfinite(logits).all())

    def test_image_preprocessing_handles_light_background_digit(self):
        image = Image.new("L", (40, 40), color=255)
        draw = ImageDraw.Draw(image)
        draw.line((12, 8, 12, 32), fill=0, width=4)
        draw.line((12, 8, 25, 8), fill=0, width=4)
        draw.line((12, 20, 23, 20), fill=0, width=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "digit.png"
            image.save(path)
            result = prepare_digit_image(path)
        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(float(result.max()), 0.5)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
