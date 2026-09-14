from vlm_camera.camera.gstreamer_pipeline import (
    jetson_csi_pipeline,
    jetson_rtsp_pipeline,
    mjpeg_http_pipeline,
)


def test_pipelines_use_nvmm_or_appsink():
    csi = jetson_csi_pipeline()
    assert "nvarguscamerasrc" in csi
    assert "NVMM" in csi
    assert "appsink" in csi
    rtsp = jetson_rtsp_pipeline("rtsp://cam/stream")
    assert "nvv4l2decoder" in rtsp
    mjpeg = mjpeg_http_pipeline("http://127.0.0.1:9090/stitch")
    assert "jpegdec" in mjpeg
