import numpy as np

from vlm_camera.vision.fisheye import crop_black_bands
from vlm_camera.vision.roi import split_triple_rois


def _letterboxed_strip():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    img[40:160, :] = (40, 80, 120)
    img[40:160, 0:80] = (20, 20, 200)  # left blob
    img[40:160, 160:240] = (20, 200, 20)
    img[40:160, 320:400] = (200, 20, 20)
    return img


def test_crop_black_bands_drops_letterbox():
    cropped = crop_black_bands(_letterboxed_strip(), luma_min=10)
    assert cropped.shape[0] < 200
    assert cropped.shape[0] >= 110
    assert cropped.shape[1] == 400


def test_split_triple_rois_keys_and_square():
    rois = split_triple_rois(_letterboxed_strip()[40:160])
    assert set(rois) == {"left", "center", "right"}
    for im in rois.values():
        assert im.shape == (384, 384, 3)
