import cv2
import numpy as np
import pytest

from pest_node.discovery import (InvalidServerUrl, decode_qr, http_base_url, normalize_server_url,
                                 scan_for_server_url)


@pytest.mark.parametrize("raw, expected", [
    ("ws://192.168.5.10:8000/ws", "ws://192.168.5.10:8000/ws"),
    ("  ws://192.168.5.10:8000/ws\n", "ws://192.168.5.10:8000/ws"),
    ("http://192.168.5.10:8000", "ws://192.168.5.10:8000/ws"),
    ("https://srv.local:8443/", "wss://srv.local:8443/ws"),
    ("WS://10.0.0.2:8000/ws", "ws://10.0.0.2:8000/ws"),
])
def test_normalize_valid(raw, expected):
    assert normalize_server_url(raw) == expected


@pytest.mark.parametrize("raw", [
    "", "   ", "192.168.5.10:8000", "ftp://h/ws", "ws://", "ws://h:99999/ws",
    "ws://user:pw@h:8000/ws", "ws://h:8000/ws?x=1", "ws://h :8000/ws", "ws://" + "a" * 600,
])
def test_normalize_invalid(raw):
    with pytest.raises(InvalidServerUrl):
        normalize_server_url(raw)


def test_http_base_url():
    assert http_base_url("ws://192.168.5.10:8000/ws") == "http://192.168.5.10:8000"
    assert http_base_url("wss://h/ws") == "https://h"


def _qr_image(text: str, scale: int = 8) -> np.ndarray:
    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    qr = cv2.copyMakeBorder(qr, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)


def test_decode_qr_roundtrip():
    assert decode_qr(_qr_image("ws://192.168.5.10:8000/ws")) == "ws://192.168.5.10:8000/ws"


def test_decode_qr_none_on_blank():
    assert decode_qr(np.full((480, 640, 3), 255, np.uint8)) is None


def test_scan_skips_invalid_then_finds_valid():
    frames = iter([_qr_image("isto nao e url"), np.zeros((100, 100, 3), np.uint8),
                   _qr_image("ws://10.1.1.1:8000/ws")])
    url = scan_for_server_url(lambda: next(frames), timeout_s=5, interval_s=0)
    assert url == "ws://10.1.1.1:8000/ws"


def test_scan_timeout():
    with pytest.raises(TimeoutError):
        scan_for_server_url(lambda: np.zeros((50, 50, 3), np.uint8), timeout_s=0.2, interval_s=0.05)
