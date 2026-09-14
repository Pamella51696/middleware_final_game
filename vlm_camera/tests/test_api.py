from fastapi.testclient import TestClient

from vlm_camera.api.server import app, get_pipeline
from vlm_camera.pipeline import SurroundPipeline


def test_health_and_analyze_jpeg():
    import cv2
    import numpy as np

    app.dependency_overrides.clear()
    # Force dummy pipeline for CI
    import vlm_camera.api.server as server

    server._pipeline = SurroundPipeline(vlm_backend="dummy")
    client = TestClient(app)
    h = client.get("/health")
    assert h.status_code == 200
    assert h.json()["ok"] is True

    img = np.zeros((80, 240, 3), dtype=np.uint8)
    img[10:70, :] = 80
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    r = client.post("/analyze", files={"file": ("f.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert "vehicles" in body
    assert body["roi_names"] == ["left", "center", "right"]
    latest = client.get("/latest")
    assert latest.status_code == 200
