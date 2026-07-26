"""
model.py
Model creation, training, evaluation and retraining logic for the Flower Vision classifier.

Architecture: MobileNetV2 (ImageNet weights, frozen base) + GlobalAveragePooling +
Dropout + Dense softmax head. This gives us a strong optimization technique
("use of a pretrained model") on top of a small dataset, plus early stopping and
regularization (dropout) during training.
"""

from pathlib import Path
import json
import datetime

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
)

IMG_SIZE = (160, 160)
NUM_CLASSES = 5
MODEL_PATH = Path("models/flower_model.h5")
MODEL_PREV_PATH = Path("models/flower_model_prev.h5")
METRICS_PATH = Path("models/metrics.json")


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def build_model(num_classes: int = NUM_CLASSES, img_size=IMG_SIZE, fine_tune: bool = False) -> tf.keras.Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = fine_tune  # frozen for initial training; True for fine-tune passes

    inputs = tf.keras.Input(shape=img_size + (3,))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# Train (from scratch / initial training)
# --------------------------------------------------------------------------
def train_model(model, train_ds, val_ds, epochs: int = 15, patience: int = 3):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=[early_stop, reduce_lr],
    )
    return history


# --------------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------------
def evaluate_model(model, test_ds, class_names):
    y_true, y_pred, y_prob = [], [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_prob.extend(probs.tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(labels.numpy().tolist())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, y_prob, labels=list(range(len(class_names)))),
    }
    cm = confusion_matrix(y_true, y_pred).tolist()
    metrics["confusion_matrix"] = cm
    metrics["class_names"] = class_names
    metrics["evaluated_at"] = datetime.datetime.utcnow().isoformat()
    return metrics


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------
def save_model(model, metrics: dict, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)


def load_model(path: Path = MODEL_PATH) -> tf.keras.Model:
    return tf.keras.models.load_model(path)


# --------------------------------------------------------------------------
# Retrain: fine-tune the EXISTING saved model on old + newly uploaded data
# --------------------------------------------------------------------------
def retrain_model(train_ds, val_ds, test_ds, class_names,
                   existing_model_path: Path = MODEL_PATH, epochs: int = 5):
    """
    Loads the current production model as the starting point (transfer/fine-tune,
    NOT training from scratch), continues training on the combined dataset, evaluates,
    and only promotes the new model if it is at least as good as the old one.
    """
    if existing_model_path.exists():
        model = load_model(existing_model_path)
        # unfreeze the base for gentle fine-tuning at a lower LR
        model.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        old_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
    else:
        model = build_model()
        old_metrics = None

    train_model(model, train_ds, val_ds, epochs=epochs, patience=2)
    new_metrics = evaluate_model(model, test_ds, class_names)

    promoted = True
    if old_metrics and new_metrics["accuracy"] < old_metrics.get("accuracy", 0):
        promoted = False

    if promoted:
        if existing_model_path.exists():
            existing_model_path.replace(MODEL_PREV_PATH)  # backup old model
        save_model(model, new_metrics, existing_model_path)

    return {
        "promoted": promoted,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
    }
