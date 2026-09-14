from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from vlm_camera.detector.detector import ObjectDetector
from vlm_camera.fusion.scene_manager import SceneManager
from vlm_camera.vision.fisheye import crop_black_bands
from vlm_camera.vision.frame_selector import FrameSelector
from vlm_camera.vision.perspective import resize_for_vlm
from vlm_camera.vision.roi import split_triple_rois
from vlm_camera.vlm.inference import VlmEngine


class SurroundPipeline:
    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        vlm_interval_sec: float = 1.0,
        vlm_backend: str = "auto",
    ):
        self.detector = ObjectDetector(weights=yolo_weights)
        self.vlm = VlmEngine(backend=vlm_backend, load=(vlm_backend != "dummy"))
        self.selector = FrameSelector(vlm_interval_sec)
        self.scene = SceneManager()

    def analyze_bgr(self, bgr: np.ndarray, force_vlm: bool = False) -> Dict[str, Any]:
        useful = crop_black_bands(bgr)
        rois = split_triple_rois(useful)
        detections = self.detector.detect(useful)
        run_vlm = force_vlm or self.selector.should_run()
        vlm_out = None
        if run_vlm:
            vlm_out = self.vlm.run(rois, detections=[d.to_dict() for d in detections])
        fused = self.scene.fuse(detections, vlm_out or self.scene.latest)
        fused["vlm_ran"] = bool(run_vlm)
        fused["roi_names"] = list(rois.keys())
        fused["useful_size"] = [int(useful.shape[1]), int(useful.shape[0])]
        fused["detector_backend"] = self.detector.backend
        return fused

    def analyze_jpeg(self, jpeg: bytes, force_vlm: bool = True) -> Dict[str, Any]:
        arr = np.frombuffer(jpeg, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return self.scene.fuse([], {"scene": "decode_error", "description": "bad jpeg"})
        return self.analyze_bgr(bgr, force_vlm=force_vlm)
