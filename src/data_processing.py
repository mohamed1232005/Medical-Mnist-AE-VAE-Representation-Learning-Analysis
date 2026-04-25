"""
data_processing.py
------------------
Handles all data loading and tf.data pipeline construction
for the Medical MNIST dataset.
"""

# Standard library imports
import os
import pathlib
from typing import Tuple

# Third-party imports
import tensorflow as tf


IMG_HEIGHT: int = 64
IMG_WIDTH: int = 64
IMG_CHANNELS: int = 1


def parse_image(file_path: str) -> tf.Tensor:
    """
    Load and preprocess a single image file.

    Args:
        file_path: Path to the image file.

    Returns:
        Preprocessed image tensor of shape (64, 64, 1), normalized to [0, 1].
    """
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=IMG_CHANNELS)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def build_dataset(
    region_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Build train and validation tf.data.Dataset pipelines
    for a given anatomical region.

    Args:
        region_path: Path to the folder containing images for this region.
        batch_size:  Number of samples per batch.
        val_split:   Fraction of data to use for validation.
        seed:        Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    patterns = [
        str(pathlib.Path(region_path) / "*.jpeg"),
        str(pathlib.Path(region_path) / "*.jpg"),
        str(pathlib.Path(region_path) / "*.png"),
    ]

    # Count total files
    all_files = tf.data.Dataset.list_files(patterns, shuffle=True, seed=seed)
    total = sum(1 for _ in all_files)
    n_val = int(total * val_split)
    n_train = total - n_val

    # Rebuild (generator was consumed)
    all_files = tf.data.Dataset.list_files(patterns, shuffle=True, seed=seed)

    # Map: input == target (reconstruction task)
    def _load_pair(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        img = parse_image(x)
        return img, img

    train_ds = (
        all_files.take(n_train)
        .map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        all_files.skip(n_train)
        .map(_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    print(f"  ✅ {os.path.basename(region_path)}: "
          f"{n_train} train | {n_val} val")
    return train_ds, val_ds


def find_region_path(dataset_root: str, region_name: str) -> str:
    """
    Search recursively for a region folder inside the dataset root.

    Args:
        dataset_root: Root directory of the downloaded dataset.
        region_name:  Name of the anatomical region folder to find.

    Returns:
        Full path to the region folder.

    Raises:
        FileNotFoundError: If the region folder is not found.
    """
    for root, dirs, _ in os.walk(dataset_root):
        for d in dirs:
            if d.lower() == region_name.lower():
                return os.path.join(root, d)
    raise FileNotFoundError(
        f"Region folder '{region_name}' not found under '{dataset_root}'"
    )