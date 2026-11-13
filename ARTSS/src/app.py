#!/usr/bin/env python3
"""
app.py - Milestone 5: FastAPI microservice for retinal DR screening (decision support, not a diagnosis).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import os
import secrets
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import transforms as T

log = logging.getLogger("artss.api")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_RETINA_FRACTION = 0.05
ALLOWED_FORMATS = {"JPEG", "PNG", "TIFF"}
DISCLAIMER = ("Decision-support output only; not a diagnosis. Results must be reviewed by a qualified clinician.")


class InputError(ValueError):
    """The uploaded image is unusable (maps to HTTP 422)."""


@dataclass(frozen=True)
class Settings:
    models_dir: Path
    tag: str | None
    api_key: str | None
    device: str | None
    max_upload_bytes: int
    max_pixels: int
    min_side: int
    max_queue: int
    timeout_s: float

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            models_dir=Path(env.get("ARTSS_MODELS_DIR") or Path.cwd() / "ARTSS" / "models").resolve(),
            tag=env.get("ARTSS_TAG") or None,
            api_key=env.get("ARTSS_API_KEY") or None,
            device=env.get("ARTSS_DEVICE") or None,
            max_upload_bytes=int(float(env.get("ARTSS_MAX_UPLOAD_MB", "15")) * 1024 * 1024),
            max_pixels=int(env.get("ARTSS_MAX_PIXELS", "40000000")),
            min_side=int(env.get("ARTSS_MIN_SIDE", "128")),
            max_queue=int(env.get("ARTSS_MAX_QUEUE", "8")),
            timeout_s=float(env.get("ARTSS_TIMEOUT_S", "60")),
        )


@dataclass(frozen=True)
class PreprocessConfig:
    size: int
    border_thresh: int = 10
    sigma_ratio: float = 1 / 30
    mask_ratio: float = 0.9
    enhance: bool = True


class FundusClassifier(nn.Module):
    def __init__(self, backbone_name: str):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


@dataclass
class Bundle:
    tag: str
    model_version: str
    backbone: str
    image_size: int
    threshold: float
    pre: PreprocessConfig
    ckpt_path: Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_tag(s: Settings) -> str:
    if s.tag:
        return s.tag
    found = sorted(p.name[: -len("_results.json")] for p in s.models_dir.glob("*_results.json"))
    if len(found) != 1:
        raise RuntimeError(f"Set ARTSS_TAG: found {len(found)} results files in {s.models_dir} ({found}).")
    return found[0]


def load_bundle(s: Settings) -> Bundle:
    tag = resolve_tag(s)
    ckpt_path = s.models_dir / f"{tag}_best.pth"
    results_path = s.models_dir / f"{tag}_results.json"
    for p in (ckpt_path, results_path):
        if not p.is_file():
            raise RuntimeError(f"Required artefact missing: {p}")

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
        cfg = data["config"]
        info = cfg["data_info"]
        backbone = str(cfg["backbone"])
        image_size = int(cfg["image_size"])
        stored_size = int(info["stored_size"])
        enhanced = bool(info["enhanced"])
        threshold = float(data["results"]["val"]["at_val_youden_threshold"]["threshold"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{results_path.name} lacks the fields this service needs ({exc!r}). "
                           "Re-run train.py (final evaluation) to regenerate it.") from exc
    if not (0.0 < threshold < 1.0):
        raise RuntimeError(f"Implausible operating threshold {threshold} in {results_path.name}.")

    pre = PreprocessConfig(size=stored_size, enhance=enhanced)
    pp_path = s.models_dir / "preprocess_config.json"
    if pp_path.is_file():
        try:
            raw = json.loads(pp_path.read_text(encoding="utf-8"))["config"]
            pre = PreprocessConfig(size=int(raw["size"]), border_thresh=int(raw["border_thresh"]),
                                   sigma_ratio=float(raw["sigma_ratio"]), mask_ratio=float(raw["mask_ratio"]),
                                   enhance=bool(raw["enhance"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unreadable {pp_path}: {exc!r}") from exc
        if (pre.size, pre.enhance) != (stored_size, enhanced):
            raise RuntimeError(f"{pp_path.name} (size={pre.size}, enhance={pre.enhance}) disagrees with the "
                               f"training run (stored_size={stored_size}, enhanced={enhanced}).")
    else:
        log.warning("%s not found: using data_prep defaults for border/sigma/mask. Copy "
                    "ARTSS/data/processed/preprocess_config.json next to the model to pin them.", pp_path.name)

    return Bundle(tag=tag, model_version=f"{tag}@{_sha256(ckpt_path)[:12]}", backbone=backbone,
                  image_size=image_size, threshold=threshold, pre=pre, ckpt_path=ckpt_path)


def find_retina_bbox(gray: np.ndarray, base_thresh: int):
    thresh = min(base_thresh, max(2, int(np.percentile(gray, 99) * 0.15)))
    mask = (cv2.GaussianBlur(gray, (5, 5), 0) > thresh).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    if mask.mean() < MIN_RETINA_FRACTION:
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1, mask


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def resize_square(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def circle_mask(size: int, ratio: float) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), int(size / 2 * ratio), 1, -1)
    return m.astype(bool)


def ben_graham(img: np.ndarray, sigma_ratio: float, inner: np.ndarray) -> np.ndarray:
    sigma = max(1.0, (img.shape[0] / 2) * sigma_ratio)
    out = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), sigma), -4, 128)
    out[~inner] = 128
    return out


def validate_upload(raw: bytes, s: Settings) -> None:
    if not raw:
        raise InputError("Empty file.")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            fmt, (w, h) = im.format, im.size
    except Exception as exc:
        raise InputError("File is not a readable image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InputError(f"Unsupported image format {fmt!r}; allowed: {sorted(ALLOWED_FORMATS)}.")
    if min(w, h) < s.min_side:
        raise InputError(f"Image too small ({w}x{h}); minimum side is {s.min_side}px.")
    if w * h > s.max_pixels:
        raise InputError(f"Image too large ({w}x{h}); maximum is {s.max_pixels} pixels.")


def preprocess(raw: bytes, cfg: PreprocessConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("Could not decode image.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = find_retina_bbox(gray, cfg.border_thresh)
    if det is None:
        raise InputError("No retinal disc detected in the uploaded image.")
    x0, y0, x1, y1, mask = det

    std = resize_square(pad_to_square(img[y0:y1, x0:x1]), cfg.size)
    inner = circle_mask(cfg.size, cfg.mask_ratio)
    blur_score = float(cv2.Laplacian(cv2.cvtColor(std, cv2.COLOR_BGR2GRAY), cv2.CV_64F)[inner].var())
    out = ben_graham(std, cfg.sigma_ratio, inner) if cfg.enhance else std

    quality = {"blur_score": round(blur_score, 2),
               "brightness": round(float(gray[mask > 0].mean()), 2),
               "retina_area_frac": round(float(mask.mean()), 4)}
    return std, cv2.cvtColor(out, cv2.COLOR_BGR2RGB), quality


class Engine:
    def __init__(self, settings: Settings, bundle: Bundle):
        self.settings, self.bundle = settings, bundle
        self.device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))

        model = FundusClassifier(bundle.backbone)
        state = torch.load(bundle.ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device, memory_format=torch.channels_last).eval()
        self.cam_layer = getattr(self.model.backbone, "conv_head", None)
        if self.cam_layer is None:
            raise RuntimeError(f"Backbone {bundle.backbone} has no conv_head for Grad-CAM; set a target layer.")

        steps = []
        if bundle.pre.size != bundle.image_size:
            steps.append(T.Resize((bundle.image_size, bundle.image_size),
                                  interpolation=T.InterpolationMode.BICUBIC, antialias=True))
        self.transform = T.Compose(steps + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.slots = asyncio.Semaphore(settings.max_queue)

        with torch.inference_mode():
            dummy = torch.zeros(1, 3, bundle.image_size, bundle.image_size, device=self.device)
            self.model(dummy.contiguous(memory_format=torch.channels_last))

    def infer(self, raw: bytes, want_cam: bool) -> dict:
        t0 = time.perf_counter()
        std_bgr, enhanced_rgb, quality = preprocess(raw, self.bundle.pre)
        tensor = self.transform(Image.fromarray(enhanced_rgb)).unsqueeze(0)
        tensor = tensor.to(self.device, memory_format=torch.channels_last)

        with torch.inference_mode():
            logit = self.model(tensor).reshape(-1)[0].float()
        prob = float(torch.sigmoid(logit))
        if not math.isfinite(prob):
            raise RuntimeError("Model produced a non-finite output.")

        positive = prob >= self.bundle.threshold
        payload = {
            "dr_probability": round(prob, 4),
            "threshold_used": round(self.bundle.threshold, 4),
            "dr_positive": bool(positive),
            "diagnosis": "DR suspected" if positive else "DR not suspected",
            "quality_metrics": quality,
            "model_version": self.bundle.model_version,
            "disclaimer": DISCLAIMER,
        }

        if want_cam:
            cam = GradCAM(model=self.model, target_layers=[self.cam_layer])
            try:
                heat = cam(input_tensor=tensor, targets=[BinaryClassifierOutputTarget(1)])[0]
            finally:
                cam.activations_and_grads.release()
            base = cv2.cvtColor(std_bgr, cv2.COLOR_BGR2RGB)
            size = self.bundle.image_size
            if base.shape[0] != size:
                base = cv2.resize(base, (size, size), interpolation=cv2.INTER_AREA)
            overlay = show_cam_on_image(base.astype(np.float32) / 255.0, heat, use_rgb=True)
            ok, buf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RuntimeError("PNG encoding of the Grad-CAM overlay failed.")
            payload["gradcam_png_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")

        payload["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return payload


class QualityMetrics(BaseModel):
    blur_score: float
    brightness: float
    retina_area_frac: float


class PredictionResponse(BaseModel):
    dr_probability: float
    threshold_used: float
    dr_positive: bool
    diagnosis: str
    quality_metrics: QualityMetrics
    model_version: str
    disclaimer: str
    latency_ms: float
    request_id: str
    gradcam_png_base64: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=os.environ.get("ARTSS_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
    settings = Settings.from_env()
    if settings.api_key is None:
        log.warning("ARTSS_API_KEY is not set: the API is UNAUTHENTICATED. Do not expose it publicly.")
    app.state.settings = settings
    app.state.engine = Engine(settings, load_bundle(settings))
    eng: Engine = app.state.engine
    log.info("Ready: %s on %s | image_size=%d stored_size=%d enhance=%s threshold=%.4f", eng.bundle.model_version,
             eng.device, eng.bundle.image_size, eng.bundle.pre.size, eng.bundle.pre.enhance, eng.bundle.threshold)
    yield
    eng.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="ARTSS Diabetic Retinopathy Screening API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("Unhandled error rid=%s %s %s", rid, request.method, request.url.path)
        response = JSONResponse(status_code=500, content={"detail": "Internal server error.", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    log.info("rid=%s %s %s -> %d in %.0fms", rid, request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000)
    return response


async def require_api_key(request: Request) -> None:
    expected = request.app.state.settings.api_key
    if expected is None:
        return
    supplied = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


async def read_limited(file: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := await file.read(1 << 20):
        buf += chunk
        if len(buf) > limit:
            raise HTTPException(status_code=413, detail=f"File exceeds the {limit // (1024 * 1024)} MB limit.")
    return bytes(buf)


@app.get("/", include_in_schema=False)
async def serve_ui():
    ui_path = Path(__file__).resolve().parent / "index.html"
    if not ui_path.is_file():
        raise HTTPException(status_code=404, detail="index.html not found next to app.py")
    return FileResponse(ui_path)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(request: Request):
    eng: Engine | None = getattr(request.app.state, "engine", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ready", "model_version": eng.bundle.model_version, "device": str(eng.device),
            "image_size": eng.bundle.image_size, "operating_threshold": eng.bundle.threshold}


@app.post("/predict", response_model=PredictionResponse, response_model_exclude_none=True,
          dependencies=[Depends(require_api_key)])
async def predict_fundus(request: Request, file: UploadFile = File(...), gradcam: bool = Query(True)):
    eng: Engine = request.app.state.engine
    raw = await read_limited(file, eng.settings.max_upload_bytes)

    if eng.slots.locked():
        raise HTTPException(status_code=503, detail="Server busy, retry shortly.", headers={"Retry-After": "2"})
    async with eng.slots:
        loop = asyncio.get_running_loop()
        try:
            validate_upload(raw, eng.settings)
            result = await asyncio.wait_for(loop.run_in_executor(eng.executor, eng.infer, raw, gradcam),
                                            timeout=eng.settings.timeout_s)
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Inference timed out.") from exc
    result["request_id"] = request.state.request_id
    return result


def parity_check(raw_path: str, processed_png: str) -> int:
    settings = Settings.from_env()
    bundle = load_bundle(settings)
    raw = Path(raw_path).read_bytes()
    _, served_rgb, _ = preprocess(raw, bundle.pre)
    with Image.open(processed_png) as im:
        trained_rgb = np.asarray(im.convert("RGB"))
    if served_rgb.shape != trained_rgb.shape:
        print(f"FAIL: shape mismatch served={served_rgb.shape} trained={trained_rgb.shape}")
        return 1
    diff = np.abs(served_rgb.astype(np.int16) - trained_rgb.astype(np.int16))
    print(f"max abs pixel diff: {int(diff.max())} | mean: {diff.mean():.4f} | differing pixels: {int((diff > 0).sum())}")
    print("PASS: serving preprocessing is pixel-identical to training." if diff.max() == 0
          else "FAIL: serving preprocessing differs from training (check OpenCV version and preprocess_config.json).")
    return 0 if diff.max() == 0 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ARTSS DR screening service")
    ap.add_argument("--parity-check", nargs=2, metavar=("RAW_IMAGE", "PROCESSED_PNG"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.parity_check:
        logging.basicConfig(level=logging.WARNING)
        sys.exit(parity_check(*a.parity_check))
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, workers=1)