import numpy as np

from vlm_camera.pipeline import SurroundPipeline
from vlm_camera.vision.frame_selector import FrameSelector
from vlm_camera.vlm.qwen_vlm import dummy_describe


def test_dummy_describe_from_detections():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    out = dummy_describe(
        [img],
        detections=[{"class": "car", "confidence": 0.9}, {"class": "person", "confidence": 0.8}],
    )
    assert out["vehicles"] == ["car"]
    assert out["people"] == ["person"]
    assert 0.0 <= out["confidence"] <= 1.0


def test_pipeline_dummy_on_synthetic_pano():
    pano = np.zeros((120, 480, 3), dtype=np.uint8)
    pano[20:100, :] = 90
    pipe = SurroundPipeline(vlm_backend="dummy")
    result = pipe.analyze_bgr(pano, force_vlm=True)
    assert result["roi_names"] == ["left", "center", "right"]
    assert result["vlm_ran"] is True
    assert "scene" in result
    assert result["useful_size"][0] == 480


def test_frame_selector_rate_limit():
    sel = FrameSelector(interval_sec=1.0)
    assert sel.should_run(0.0) is True
    assert sel.should_run(0.2) is False
    assert sel.should_run(1.05) is True
