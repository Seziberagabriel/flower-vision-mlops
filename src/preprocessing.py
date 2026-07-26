"""
preprocessing.py
Data acquisition + preprocessing utilities for the Flower Vision pipeline.

Responsibilities:
    - Download & extract the raw flower_photos dataset
    - Build a stratified train/test split on disk (data/train, data/test)
    - Decode / resize / normalize images for model input
    - Build tf.data pipelines (with augmentation) for training
    - Prepare newly bulk-uploaded images for a retraining round
"""

import os
import io
import shutil
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

DATA_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
RAW_DIR = Path("data/raw")
TRAIN_DIR = Path("data/train")
TEST_DIR = Path("data/test")


# --------------------------------------------------------------------------
# 1. Data acquisition
# --------------------------------------------------------------------------
def download_dataset(dest_dir: Path = RAW_DIR) -> Path:
    """Download and extract the flower_photos dataset if not already present."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "flower_photos.tgz"
    extracted_path = dest_dir / "flower_photos"

    if extracted_path.exists():
        print(f"Dataset already present at {extracted_path}")
        return extracted_path

    print("Downloading flower_photos.tgz ...")
    urllib.request.urlretrieve(DATA_URL, archive_path)

    print("Extracting ...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(dest_dir)

    return extracted_path


# --------------------------------------------------------------------------
# 2. Train / test split
# --------------------------------------------------------------------------
def build_train_test_split(source_dir: Path, test_ratio: float = 0.15, seed: int = 42):
    """Copy images from the raw per-class folders into data/train/<class> and data/test/<class>."""
    random.seed(seed)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for class_name in CLASS_NAMES:
        class_src = source_dir / class_name
        if not class_src.exists():
            continue

        images = [f for f in class_src.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        random.shuffle(images)

        n_test = max(1, int(len(images) * test_ratio))
        test_images = images[:n_test]
        train_images = images[n_test:]

        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (TEST_DIR / class_name).mkdir(parents=True, exist_ok=True)

        for f in train_images:
            shutil.copy(f, TRAIN_DIR / class_name / f.name)
        for f in test_images:
            shutil.copy(f, TEST_DIR / class_name / f.name)

    print(f"Train/test split ready under {TRAIN_DIR} and {TEST_DIR}")


# --------------------------------------------------------------------------
# 3. tf.data pipelines
# --------------------------------------------------------------------------
def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    return image, label


def get_datasets(train_dir: Path = TRAIN_DIR, test_dir: Path = TEST_DIR,
                  img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_split=0.2, seed=42):
    """Return (train_ds, val_ds, test_ds, class_names) as tf.data.Dataset objects."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="training", seed=seed,
        image_size=img_size, batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="validation", seed=seed,
        image_size=img_size, batch_size=batch_size,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=img_size, batch_size=batch_size, shuffle=False,
    )

    class_names = train_ds.class_names

    train_ds = train_ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    normalize = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1)  # MobileNetV2 preprocessing range
    train_ds = train_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (normalize(x), y)).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# --------------------------------------------------------------------------
# 4. Single-image preprocessing (for inference)
# --------------------------------------------------------------------------
def preprocess_image_bytes(image_bytes: bytes, img_size=IMG_SIZE) -> np.ndarray:
    """Decode raw uploaded image bytes into a model-ready batch of shape (1, H, W, 3)."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(img_size)
    arr = np.array(image).astype("float32")
    arr = (arr / 127.5) - 1.0  # match MobileNetV2 preprocessing used in training
    return np.expand_dims(arr, axis=0)


# --------------------------------------------------------------------------
# 5. Prepare a bulk-uploaded batch for retraining
# --------------------------------------------------------------------------
def prepare_retrain_batch(upload_dir: Path, train_dir: Path = TRAIN_DIR):
    """
    Move newly uploaded, class-labeled images (upload_dir/<class>/*.jpg) into the
    training directory so the next retraining run picks them up. Returns count added.
    """
    added = 0
    upload_dir = Path(upload_dir)
    for class_dir in upload_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        target_dir = train_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in class_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.copy(f, target_dir / f.name)
                added += 1
    return added


if __name__ == "__main__":
    raw = download_dataset()
    build_train_test_split(raw)
