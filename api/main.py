"""
api/main.py
FastAPI service for the Flower Vision pipeline.

Endpoints
---------
GET  /status                -> uptime, model version/last-retrained info, pending-upload count
POST /predict                -> upload ONE image, get back predicted class + confidence
POST /upload                 -> bulk upload a ZIP of class-labeled images for retraining
POST /retrain                -> trigger a retraining job on all uploaded-but-not-yet-trained data
GET  /visualizations         -> JSON data powering the UI's charts (class distribution etc.)
"""

import io
import os
import sys
import time
import zipfile
import shutil
import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow `import src.*`

from src.preprocessing import CLASS_NAMES, get_datasets, prepare_retrain_batch, TRAIN_DIR, TEST_DIR
from src.model import retrain_model, MODEL_PATH, METRICS_PATH
from src.prediction import predict_image_bytes

APP_START_TIME = time.time()
UPLOAD_DIR = Path("data/uploads")
RETRAIN_STATE_PATH = Path("models/retrain_state.json")

app = FastAPI(title="Flower Vision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

retrain_job_state = {"status": "idle", "last_run": None, "detail": None}


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict


@app.get("/status")
def status():
    uptime_seconds = time.time() - APP_START_TIME
    metrics = {}
    if METRICS_PATH.exists():
        import json
        metrics = json.loads(METRICS_PATH.read_text())

    pending = sum(1 for _ in UPLOAD_DIR.rglob("*.*")) if UPLOAD_DIR.exists() else 0

    return {
        "status": "healthy" if MODEL_PATH.exists() else "no model loaded",
        "uptime_seconds": round(uptime_seconds, 1),
        "model_last_trained": metrics.get("evaluated_at"),
        "model_accuracy": metrics.get("accuracy"),
        "pending_uploaded_images": pending,
        "retrain_job": retrain_job_state,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not yet trained/deployed.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    try:
        result = predict_image_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
    return result


@app.post("/upload")
async def upload_bulk(file: UploadFile = File(...)):
    """
    Accepts a ZIP file structured as:
        upload.zip
        ├── daisy/*.jpg
        ├── roses/*.jpg
        └── ...
    Saves images under data/uploads/<class>/ ready for the next retraining run.
    """
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip of class-labeled folders.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    contents = await file.read()

    try:
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            zf.extractall(UPLOAD_DIR)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")

    added = sum(1 for _ in UPLOAD_DIR.rglob("*.*"))

    # log it, acting as our lightweight "database" record of pending uploads
    log_path = Path("data/uploads_log.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{datetime.datetime.utcnow().isoformat()},{file.filename},{added}\n")

    return {"message": "Upload received.", "total_pending_images": added}


def _run_retraining_job():
    retrain_job_state["status"] = "running"
    retrain_job_state["detail"] = None
    try:
        added = prepare_retrain_batch(UPLOAD_DIR, TRAIN_DIR)
        train_ds, val_ds, test_ds, class_names = get_datasets()
        result = retrain_model(train_ds, val_ds, test_ds, class_names)

        # clear the upload staging area once folded into train/
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        retrain_job_state["status"] = "completed"
        retrain_job_state["last_run"] = datetime.datetime.utcnow().isoformat()
        retrain_job_state["detail"] = {
            "images_added": added,
            "promoted": result["promoted"],
            "new_accuracy": result["new_metrics"]["accuracy"],
            "old_accuracy": (result["old_metrics"] or {}).get("accuracy"),
        }
    except Exception as e:
        retrain_job_state["status"] = "failed"
        retrain_job_state["detail"] = str(e)


@app.post("/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    if retrain_job_state["status"] == "running":
        raise HTTPException(status_code=409, detail="A retraining job is already running.")
    if not UPLOAD_DIR.exists() or not any(UPLOAD_DIR.rglob("*.*")):
        raise HTTPException(status_code=400, detail="No newly uploaded data to retrain on. Upload data first.")

    background_tasks.add_task(_run_retraining_job)
    retrain_job_state["status"] = "queued"
    return {"message": "Retraining job started in the background. Poll /status for progress."}


@app.get("/visualizations")
def visualizations():
    """Returns JSON summaries the UI turns into charts: class distribution, image sizes, etc."""
    import json
    from PIL import Image

    class_counts = {}
    avg_brightness = {}
    sample_sizes = []

    if TRAIN_DIR.exists():
        for class_dir in TRAIN_DIR.iterdir():
            if not class_dir.is_dir():
                continue
            files = list(class_dir.glob("*.*"))
            class_counts[class_dir.name] = len(files)

            brightness_vals = []
            for f in files[:25]:  # sample for speed
                try:
                    img = Image.open(f).convert("L")
                    brightness_vals.append(sum(img.getdata()) / (img.width * img.height))
                    sample_sizes.append(img.size)
                except Exception:
                    continue
            avg_brightness[class_dir.name] = (
                sum(brightness_vals) / len(brightness_vals) if brightness_vals else None
            )

    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())

    return {
        "class_distribution": class_counts,
        "average_brightness_per_class": avg_brightness,
        "model_metrics": metrics,
    }
