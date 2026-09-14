from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from vlm_camera.pipeline import SurroundPipeline

app = FastAPI(title="Orin surround VLM", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Optional[SurroundPipeline] = None


def get_pipeline() -> SurroundPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SurroundPipeline(
            yolo_weights=os.environ.get("YOLO_WEIGHTS"),
            vlm_interval_sec=float(os.environ.get("VLM_INTERVAL_SEC", "1.0")),
            vlm_backend=os.environ.get("VLM_BACKEND", "dummy"),
        )
    return _pipeline


class AnalyzeResponse(BaseModel):
    vehicles: list
    people: list
    hazards: list
    scene: str
    description: str
    confidence: float
    objects: list = []
    hazard: bool = False
    backend: str = "dummy"


@app.get("/health")
def health():
    p = get_pipeline()
    return {
        "ok": True,
        "vlm_backend": p.vlm.qwen.backend if p.vlm.qwen.available() else "dummy",
        "detector": p.detector.backend,
    }


@app.get("/latest")
def latest():
    return get_pipeline().scene.latest


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    jpeg = await file.read()
    return get_pipeline().analyze_jpeg(jpeg, force_vlm=True)
