"""Hardware-accelerated GStreamer paths for Jetson (NVMM)."""


def jetson_csi_pipeline(
    sensor_id: int = 0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> str:
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1"
    )


def jetson_rtsp_pipeline(url: str) -> str:
    return (
        f"rtspsrc location={url} latency=100 ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! "
        "appsink drop=1"
    )


def mjpeg_http_pipeline(url: str) -> str:
    """Pull the Java server's /stitch MJPEG stream (CPU decode is fine)."""
    return (
        f"souphttpsrc location={url} is-live=true ! "
        "multipartdemux ! jpegdec ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
    )
