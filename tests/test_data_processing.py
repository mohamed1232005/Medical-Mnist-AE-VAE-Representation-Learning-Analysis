"""
test_data_processing.py
-----------------------
Unit tests for data_processing.py functions.
"""

# Standard library imports
import os
import tempfile
import unittest

# Third-party imports
import numpy as np
import tensorflow as tf
from PIL import Image

# Local imports
from src.data_processing import parse_image, build_dataset, find_region_path


class TestParseImage(unittest.TestCase):
    """Tests for the parse_image function."""

    def setUp(self) -> None:
        """Create a temporary dummy JPEG image for testing."""
        self.tmp_dir = tempfile.mkdtemp()
        img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        img = Image.fromarray(img_array, mode="L")
        self.img_path = os.path.join(self.tmp_dir, "test.jpeg")
        img.save(self.img_path)

    def test_output_shape(self) -> None:
        """parse_image should return a (64, 64, 1) tensor."""
        tensor = parse_image(self.img_path)
        self.assertEqual(tensor.shape, (64, 64, 1))

    def test_pixel_range(self) -> None:
        """Pixel values should be normalized to [0, 1]."""
        tensor = parse_image(self.img_path)
        self.assertGreaterEqual(float(tf.reduce_min(tensor)), 0.0)
        self.assertLessEqual(float(tf.reduce_max(tensor)), 1.0)

    def test_dtype(self) -> None:
        """Output tensor should be float32."""
        tensor = parse_image(self.img_path)
        self.assertEqual(tensor.dtype, tf.float32)


class TestBuildDataset(unittest.TestCase):
    """Tests for the build_dataset function."""

    def setUp(self) -> None:
        """Create a temporary directory with dummy JPEG images."""
        self.tmp_dir = tempfile.mkdtemp()
        for i in range(20):
            img_array = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            img = Image.fromarray(img_array, mode="L")
            img.save(os.path.join(self.tmp_dir, f"img_{i:03d}.jpeg"))

    def test_returns_two_datasets(self) -> None:
        """build_dataset should return a tuple of two tf.data.Dataset."""
        train_ds, val_ds = build_dataset(self.tmp_dir, batch_size=4)
        self.assertIsInstance(train_ds, tf.data.Dataset)
        self.assertIsInstance(val_ds, tf.data.Dataset)

    def test_batch_shape(self) -> None:
        """Each batch should have shape (batch, 64, 64, 1)."""
        train_ds, _ = build_dataset(self.tmp_dir, batch_size=4)
        for x, y in train_ds.take(1):
            self.assertEqual(x.shape[-3:], (64, 64, 1))
            self.assertEqual(y.shape[-3:], (64, 64, 1))

    def test_input_equals_target(self) -> None:
        """Input and target should be identical (reconstruction task)."""
        train_ds, _ = build_dataset(self.tmp_dir, batch_size=4)
        for x, y in train_ds.take(1):
            np.testing.assert_array_equal(x.numpy(), y.numpy())


class TestFindRegionPath(unittest.TestCase):
    """Tests for the find_region_path function."""

    def setUp(self) -> None:
        """Create a temporary directory tree simulating dataset structure."""
        self.root = tempfile.mkdtemp()
        self.region_dir = os.path.join(self.root, "AbdomenCT")
        os.makedirs(self.region_dir)

    def test_finds_existing_region(self) -> None:
        """Should find an existing region folder."""
        path = find_region_path(self.root, "AbdomenCT")
        self.assertEqual(path, self.region_dir)

    def test_case_insensitive(self) -> None:
        """Search should be case-insensitive."""
        path = find_region_path(self.root, "abdomenct")
        self.assertEqual(path, self.region_dir)

    def test_raises_for_missing_region(self) -> None:
        """Should raise FileNotFoundError for missing regions."""
        with self.assertRaises(FileNotFoundError):
            find_region_path(self.root, "NonExistentRegion")


if __name__ == "__main__":
    unittest.main()