"""
prediction.py
Single-image inference helper used by both the API and the notebook.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf

from src.preprocessing import preprocess_image_bytes, CLASS_NAMES
from src.model import MODEL_PATH

_model_cache = {"model": None, "mtime": None}


def get_model() -> tf.keras.Model:
    """Lazily load (and hot-reload after retraining) the production model."""
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    if _model_cache["model"] is None or _model_cache["mtime"] != mtime:
        _model_cache["model"] = tf.keras.models.load_model(MODEL_PATH)
        _model_cache["mtime"] = mtime
    return _model_cache["model"]


def predict_image_bytes(image_bytes: bytes) -> dict:
    """Run inference on a single raw image (bytes) and return label + confidence scores."""
    model = get_model()
    batch = preprocess_image_bytes(image_bytes)
    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "all_probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }
