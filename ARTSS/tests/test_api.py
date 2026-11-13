#!/usr/bin/env python3
"""
test_api.py - Automated integration test suite for the ARTSS FastAPI microservice.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure ARTSS/src is importable
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers FastAPI's lifespan (loads the model)
    with TestClient(app) as c:
        yield c


def _make_synthetic_fundus_png(size: int = 512, black_only: bool = False) -> bytes:
    """Creates an in-memory PNG (either all-black or with a synthetic bright retinal disc)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    if not black_only:
        yy, xx = np.ogrid[:size, :size]
        center = size // 2
        radius = int(size * 0.4)
        mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
        img[mask] = [180, 95, 45]  # Synthetic orange-red fundus disc
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_ready(client: TestClient):
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    body = r_ready.json()
    assert body["status"] == "ready"
    assert 0.0 < body["operating_threshold"] < 1.0
    assert body["image_size"] == 224


def test_ui_served(client: TestClient):
    r_ui = client.get("/")
    assert r_ui.status_code == 200
    assert "ARTSS" in r_ui.text


def test_rejects_empty_and_corrupt_files(client: TestClient):
    r_empty = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert r_empty.status_code == 422

    r_bad = client.post("/predict", files={"file": ("corrupt.png", b"not-an-image", "image/png")})
    assert r_bad.status_code == 422


def test_rejects_blank_black_image_without_retina(client: TestClient):
    black_png = _make_synthetic_fundus_png(size=256, black_only=True)
    r = client.post("/predict", files={"file": ("black.png", black_png, "image/png")})
    assert r.status_code == 422
    assert "No retinal disc" in r.json()["detail"]


def test_end_to_end_predict_with_gradcam(client: TestClient):
    raw_sample = Path("ARTSS/data/raw/APTOS-2019/train_images/train_images/1ae8c165fd53.png")
    payload_bytes = raw_sample.read_bytes() if raw_sample.is_file() else _make_synthetic_fundus_png()

    r = client.post("/predict?gradcam=true", files={"file": ("scan.png", payload_bytes, "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["dr_probability"] <= 1.0
    assert isinstance(data["dr_positive"], bool)
    assert "gradcam_png_base64" in data and len(data["gradcam_png_base64"]) > 100
    assert "blur_score" in data["quality_metrics"]