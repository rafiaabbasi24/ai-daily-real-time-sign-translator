import unittest
import numpy as np
import torch
import cv2

# The project is expected to expose a `SignLanguageTranslator` class in the
# `src.main` module.  Adjust the import path if the actual location differs.
try:
    from src.main import SignLanguageTranslator
except ImportError as e:
    raise ImportError(
        "Unable to import `SignLanguageTranslator` from `src.main`. "
        "Make sure the project structure matches the test expectations."
    ) from e


class TestSignLanguageTranslator(unittest.TestCase):
    """Unit tests for the SignLanguageTranslator core functionality."""

    @classmethod
    def setUpClass(cls):
        """Create a dummy image and a translator instance for all tests."""
        # A plain black image (480x640) – the exact size is not crucial because
        # the translator should internally resize/crop as needed.
        cls.dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Instantiate the translator.  If the class requires a model path,
        # the default constructor should handle a fallback (e.g., a bundled
        # lightweight model).  Adjust the argument list if the signature
        # differs.
        cls.translator = SignLanguageTranslator()

        # Load the model once for all tests; this also validates that the
        # loading routine works without raising.
        cls.model = cls.translator.load_model()
        assert isinstance(
            cls.model, torch.nn.Module
        ), "load_model() must return a torch.nn.Module instance"

    def test_preprocess_frame_returns_tensor(self):
        """preprocess_frame should convert an image to a normalized torch tensor."""
        tensor = self.translator.preprocess_frame(self.dummy_image)

        # Verify type and basic shape expectations.
        self.assertIsInstance(tensor, torch.Tensor, "Output must be a torch.Tensor")
        # Expect a batch dimension (1) followed by C, H, W.
        self.assertEqual(tensor.dim(), 4, "Tensor must be 4‑dimensional (B, C, H, W)")
        self.assertEqual(tensor.shape[0], 1, "Batch dimension should be 1")
        self.assertEqual(tensor.shape[1], 3, "Channel dimension should be 3 (RGB)")

        # Values should be in the range [0, 1] after normalization.
        self.assertTrue(
            torch.all(tensor >= 0) and torch.all(tensor <= 1),
            "Tensor values should be normalized between 0 and 1",
        )

    def test_predict_returns_single_letter(self):
        """predict should return a single alphabetic character for a valid frame."""
        tensor = self.translator.preprocess_frame(self.dummy_image)
        result = self.translator.predict(tensor)

        # The result must be a string representing one letter.
        self.assertIsInstance(result, str, "Prediction must be a string")
        self.assertEqual(len(result), 1, "Prediction should be a single character")
        self.assertTrue(
            result.isalpha(),
            "Prediction should be an alphabetic character",
        )

    def test_predict_raises_on_invalid_input(self):
        """predict should raise a ValueError when given an incorrectly shaped tensor."""
        # Create a tensor with an unexpected shape (e.g., missing batch dimension).
        invalid_tensor = torch.randn(3, 224, 224)  # shape (C, H, W) instead of (B, C, H, W)

        with self.assertRaises(ValueError):
            self.translator.predict(invalid_tensor)

    def test_load_model_is_idempotent(self):
        """Calling load_model multiple times should return the same model instance."""
        model_first = self.translator.load_model()
        model_second = self.translator.load_model()
        self.assertIs(
            model_first,
            model_second,
            "load_model should return the same model instance on subsequent calls",
        )

    def test_end_to_end_pipeline(self):
        """Run a full inference pass: preprocess → predict."""
        # This test ensures that the two steps work together without error.
        tensor = self.translator.preprocess_frame(self.dummy_image)
        prediction = self.translator.predict(tensor)

        self.assertIsInstance(prediction, str)
        self.assertEqual(len(prediction), 1)
        self.assertTrue(prediction.isalpha())


if __name__ == "__main__":
    unittest.main()