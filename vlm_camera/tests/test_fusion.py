from vlm_camera.detector.detector import DetectedObject
from vlm_camera.fusion.scene_manager import SceneManager


def test_fuse_fills_vehicles_from_detector():
    mgr = SceneManager()
    dets = [
        DetectedObject("truck", 0.9, [1, 2, 3, 4]),
        DetectedObject("person", 0.8, [5, 6, 7, 8]),
    ]
    out = mgr.fuse(dets, {"scene": "street", "description": "test", "confidence": 0.4})
    assert "truck" in out["vehicles"]
    assert "person" in out["people"]
    assert out["objects"][0]["class"] == "truck"
    assert out["recent_vehicles"] == ["truck"]
    assert mgr.latest["scene"] == "street"
