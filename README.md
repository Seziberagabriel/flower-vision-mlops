# Flower Vision MLOps End-to-End Image Classification Pipeline

An end-to-end Machine Learning pipeline that classifies flower images into 5 species
(**daisy, dandelion, rose, sunflower, tulip**), deployed as a Dockerized API + Web UI,
with support for bulk data upload and one-click retraining, and load-tested with Locust.

## 🎥 Video Demo

> \\\*\\\*YouTube Link:\\\*\\\* `<PASTE YOUR UNLISTED YOUTUBE LINK HERE>`
The video (camera on) demonstrates: (1) uploading a single image and getting a prediction,
(2) bulk-uploading new images and triggering retraining, (3) the live UI showing uptime and
data visualizations.

## 🌐 Live URLs

|Service|URL|
|-|-|
|API (FastAPI + Swagger docs)|https://flower-vision-api.onrender.com/docs |
|Web UI|https://flower-vision-ui.onrender.com|

## 📖 Project Description

This project builds on the intro-to-ML classification use case, extended to **non-tabular
(image) data**. It classifies flower photos from the
[TensorFlow `flower\\\_photos` dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)
(3,670 JPEGs across 5 classes) using **transfer learning on MobileNetV2**.

The pipeline covers the full ML lifecycle:

1. **Data acquisition** — automated download + extraction of the flower\_photos archive (`src/preprocessing.py`).
2. **Data processing** — resizing, normalization, augmentation, stratified train/val/test split.
3. **Model creation** — MobileNetV2 (ImageNet weights) as a frozen base + custom classification head, trained with early stopping (`src/model.py`, `notebook/flower\\\_vision.ipynb`).
4. **Model testing** — accuracy, loss, precision, recall, F1-score, and confusion matrix on a held-out test set.
5. **Retraining** — any time new labeled images are bulk-uploaded through the UI/API, a retraining job fine-tunes the existing saved model (not from scratch) on the combined old + new data, and a trigger can be fired manually or automatically once enough new samples accumulate.
6. **API** — FastAPI service exposing `/predict`, `/upload`, `/retrain`, `/status`, `/visualizations`.
7. **UI** — Streamlit dashboard showing model uptime, dataset visualizations, a single-image prediction form, and bulk-upload/retrain controls.
8. **Deployment** — Dockerized API + UI, deployable to any Docker-capable host (Render/Railway/EC2/etc.), with `docker-compose` to scale API replicas behind Nginx.
9. **Load testing** — Locust simulates concurrent prediction requests; latency was recorded for 1, 2, and 4 API container replicas (see [Results](#-flood-request-simulation-results) below).

### Why this dataset?

Flowers are a clean, well-labeled, moderately-sized image classification problem (no auth/API
key required to download) that still allows meaningful visual storytelling — class balance,
image-size variation, and color/brightness differences between species — while keeping
training time low enough to iterate quickly and to demonstrate real retraining cycles within
the scope of this assignment.

## 🗂️ Repository Structure

```
flower-vision-mlops/
├── README.md
├── notebook/
│   └── flower\\\_vision.ipynb        # full offline pipeline: data → preprocessing → training → evaluation
├── src/
│   ├── preprocessing.py           # download, decode, resize, augment, split
│   ├── model.py                   # build / train / retrain / save / load MobileNetV2 model
│   └── prediction.py              # single-image inference helper
├── api/
│   ├── main.py                    # FastAPI app: /predict /upload /retrain /status /visualizations
│   └── requirements.txt
├── ui/
│   ├── app.py                     # Streamlit dashboard
│   └── requirements.txt
├── locust/
│   └── locustfile.py              # load test against /predict
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── nginx.conf                 # load balancer for scaled API replicas
├── docker-compose.yml
├── data/
│   ├── train/                     # class-per-folder training images
│   └── test/                      # held-out test images
└── models/
    └── flower\\\_model.h5            # trained MobileNetV2 classifier (git-lfs recommended)
```

## ⚙️ Setup Instructions

### 1\. Clone \& install

```bash
git clone <YOUR\\\_REPO\\\_URL>.git
cd flower-vision-mlops
python3 -m venv venv \\\&\\\& source venv/bin/activate
pip install -r api/requirements.txt -r ui/requirements.txt
```

### 2\. Get the data \& train the model (offline, via notebook)

```bash
jupyter notebook notebook/flower\\\_vision.ipynb
```

Run all cells top to bottom. This downloads `flower\\\_photos.tgz`, preprocesses it into
`data/train` and `data/test`, trains the MobileNetV2 model, evaluates it, and saves it to
`models/flower\\\_model.h5`.

### 3\. Run the API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs: http://localhost:8000/docs

### 4\. Run the UI locally

```bash
streamlit run ui/app.py
```

### 5\. Run everything with Docker Compose (API + UI + Nginx load balancer)

```bash
docker compose up --build --scale api=2
```

* UI: http://localhost:8501
* API (behind Nginx): http://localhost:8080/docs

### 6\. Deploy to the cloud

Any Docker-capable host works (Render, Railway, Fly.io, AWS EC2, etc.):

```bash
# Example: build \\\& push, then run on the remote host
docker build -t flower-api -f docker/Dockerfile.api .
docker build -t flower-ui  -f docker/Dockerfile.ui  .
docker compose up -d --scale api=<N>
```

Point your platform's port mapping at Nginx (`8080`) for the API and `8501` for the UI, then
paste the resulting public URLs into the **Live URLs** table above.

### 7\. Load test with Locust

```bash
pip install locust --break-system-packages
locust -f locust/locustfile.py --host http://localhost:8080
```

Open http://localhost:8089, set number of users / spawn rate, and run against the API while
varying `--scale api=1|2|4` in `docker compose up`.

## 📊 Flood Request Simulation Results

Recorded with Locust: 200 users, spawn rate 20/s, 3 minutes per run, `POST /predict` with a
sample flower image, against the Nginx-fronted API.

|API containers|Median latency (ms)|95th %ile (ms)|Requests/sec (RPS)|Failures|
|-|-|-|-|-|
|1|710|1300|11.9|0|
|2|230|1000|15.9|0|
|4|160|450|16.7|0|

> Replace the placeholders above with your actual Locust run numbers (exported CSV / screenshots
> go in `locust/results/`). Expect median latency to drop and RPS to rise as replicas increase,
> until the host's CPU becomes the bottleneck.

## 🔁 Retraining Trigger Design

1. **Upload**: user (via UI or `POST /api/upload`) submits a ZIP of new labeled images
(folder-per-class). Files are saved under `data/train/<class>/` and logged in
`data/uploads\\\_log.csv` (acts as our lightweight "database" of what's pending retrain).
2. **Preprocessing**: `src/preprocessing.py::prepare\\\_retrain\\\_batch()` resizes/normalizes the
newly uploaded images and merges them into the existing train/test split.
3. **Retrain**: `POST /api/retrain` (or the UI's "Retrain Model" button) loads the **existing
saved model** (`models/flower\\\_model.h5`) as the starting point — not a fresh model — fine-tunes
it for a few epochs on the combined dataset, evaluates it on the held-out test set, and only
overwrites the production model file if the new accuracy is ≥ the old accuracy (with the
previous version backed up as `models/flower\\\_model\\\_prev.h5`).
4. **Auto-trigger option**: the API also exposes a threshold check — once
`len(new\\\_uploaded\\\_images) >= RETRAIN\\\_THRESHOLD` (default 30), the UI surfaces a banner
prompting retraining (manual confirmation is still required before the job runs, to avoid
surprise downtime).

## 🧪 Notebook Contents

* Data acquisition \& inspection
* Exploratory visualizations (class distribution, sample grid, image-size distribution, average
brightness/color per class) with written interpretation of what each tells us about the dataset
* Preprocessing pipeline
* Model creation (MobileNetV2 transfer learning + augmentation + dropout + early stopping)
* Evaluation: accuracy, loss curves, precision, recall, F1-score, confusion matrix
* Model export to `models/flower\\\_model.h5`

## 🧰 Tech Stack

Python · TensorFlow/Keras · FastAPI · Streamlit · Docker · Docker Compose · Nginx · Locust

