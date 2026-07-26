"""
locust/locustfile.py
Simulates a flood of prediction requests against the deployed Flower Vision API.

Usage:
    locust -f locust/locustfile.py --host http://localhost:8080

Then open http://localhost:8089, set (users, spawn rate), start the run, and vary
the number of API replicas between runs via:
    docker compose up --build --scale api=1
    docker compose up --build --scale api=2
    docker compose up --build --scale api=4
Record the median/95th-percentile latency and RPS reported by Locust's UI/CSV export
for each replica count into the README's results table.
"""

import os
from pathlib import Path
from locust import HttpUser, task, between

# A small sample image bundled with the repo for load testing.
SAMPLE_IMAGE_PATH = Path(__file__).parent / "sample_flower.jpg"


class FlowerVisionUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        if SAMPLE_IMAGE_PATH.exists():
            self.image_bytes = SAMPLE_IMAGE_PATH.read_bytes()
        else:
            # 1x1 pixel fallback so the load test still runs even without a sample image;
            # replace sample_flower.jpg with a real flower photo for a realistic test.
            self.image_bytes = (
                b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                b"\xff\xdb\x00\x43\x00" + b"\x08" * 64 + b"\xff\xd9"
            )

    @task
    def predict(self):
        files = {"file": ("sample_flower.jpg", self.image_bytes, "image/jpeg")}
        self.client.post("/predict", files=files, name="/predict")

    @task(1)
    def status(self):
        self.client.get("/status", name="/status")
