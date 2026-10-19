from pathlib import Path

import cv2
import numpy as np
import pytest

from pest_node.alerts import AlertPolicy
from pest_node.camera import CameraError, FakeCamera, PiCamera
from pest_node.models import Detection


def _write_imgs(tmp_path: Path, n=3):
    for i in range(n):
        cv2.imwrite(str(tmp_path / f"img{i}.jpg"), np.full((480, 640, 3), i * 50, np.uint8))
    (tmp_path / "leia-me.txt").write_text("ignorar")


def test_fake_camera_cycles_and_reuses_buffer(tmp_path):
    _write_imgs(tmp_path)
    cam = FakeCamera(tmp_path, 1296, 972)
    cam.start()
    f1 = cam.capture()
    f2 = cam.capture()
    assert f1.shape == (972, 1296, 3) and f1.dtype == np.uint8
    assert f1 is f2  # mesmo buffer: sem acumular quadros
    names = [cam.capture() is not None and cam.current_file.name for _ in range(3)]
    assert names == ["img2.jpg", "img0.jpg", "img1.jpg"]


def test_fake_camera_errors(tmp_path):
    with pytest.raises(CameraError):
        FakeCamera(tmp_path / "nao-existe").start()
    with pytest.raises(CameraError):
        FakeCamera(tmp_path).start()


def test_picamera_without_library_gives_clear_error():
    try:
        import picamera2  # noqa: F401
        pytest.skip("Picamera2 instalado (rodando no Pi)")
    except ImportError:
        pass
    with pytest.raises(CameraError, match="python3-picamera2"):
        PiCamera().start()


def det(name, conf):
    return Detection(0, name, conf, (0, 0, 1, 1))


def test_alert_threshold_and_cooldown():
    pol = AlertPolicy(threshold=0.6, cooldown_s=60)
    assert pol.triggering([det("Percevejo", 0.5)], now=0) == []
    hits = pol.triggering([det("Percevejo", 0.7), det("Percevejo", 0.9), det("Caracol", 0.65)], now=1)
    assert sorted((d.class_name, d.confidence) for d in hits) == [("Caracol", 0.65), ("Percevejo", 0.9)]
    assert pol.triggering([det("Percevejo", 0.95)], now=30) == []  # em cooldown
    assert len(pol.triggering([det("Percevejo", 0.95)], now=62)) == 1


def test_alert_class_filter():
    pol = AlertPolicy(threshold=0.5, cooldown_s=0, classes=["Caracol"])
    assert pol.triggering([det("Percevejo", 0.99)], now=0) == []
    assert len(pol.triggering([det("Caracol", 0.6)], now=0)) == 1
