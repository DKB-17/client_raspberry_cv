"""Testes do detector.

O teste de fidelidade compara a saída deste detector (ONNX Runtime + numpy)
com os JSONs que o próprio servidor_cv gravou ao processar as mesmas imagens
com Ultralytics (servidor_cv/data/<data>/classified|unclassified/<id>.json).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pest_node.detector import (Detector, ModelError, detect_layout, letterbox, letterbox_params, nms,
                                parse_class_names, postprocess, scale_boxes)

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "teste2.onnx"
SERVER_DATA = ROOT.parent / "servidor_cv" / "data"

EXPECTED_CLASSES = [
    "Lagarta-da-soja", "Joaninha", "Gafanhoto", "Lagartas do grupo Heliothinae", "Eriopsis connexa",
    "Percevejo", "Percevejo-Marrom", "Percevejo-verde-ninfa", "Caracol", "Besouro Tenebrionidae",
    "Percevejo-verde", "Percevejo-verde-ninfa-adulta",
]


# ----------------------------- letterbox ---------------------------------- #
@pytest.mark.parametrize("h, w, exp", [
    (972, 1296, (640 / 1296, 640, 480, 0, 80)),
    (960, 540, (640 / 960, 360, 640, 140, 0)),
    (640, 640, (1.0, 640, 640, 0, 0)),
    (480, 640, (1.0, 640, 480, 0, 80)),
])
def test_letterbox_params(h, w, exp):
    r, nw, nh, left, top = letterbox_params(h, w, 640)
    assert (nw, nh, left, top) == exp[1:]
    assert r == pytest.approx(exp[0])


def test_letterbox_fills_gray_and_reuses_buffer():
    img = np.full((972, 1296, 3), 10, np.uint8)
    buf = np.empty((640, 640, 3), np.uint8)
    out, r, left, top = letterbox(img, 640, out=buf)
    assert out is buf
    assert (out[:80] == 114).all() and (out[560:] == 114).all()
    assert (out[80:560] == 10).all()


def test_scale_boxes_roundtrip():
    h, w = 972, 1296
    r, _, _, left, top = letterbox_params(h, w, 640)
    orig = np.array([[100.0, 200.0, 500.0, 700.0]], np.float32)
    in_lb = orig * r + np.array([left, top, left, top], np.float32)
    back = scale_boxes(in_lb.copy(), r, left, top, h, w)
    np.testing.assert_allclose(back, orig, atol=1e-3)


def test_scale_boxes_clips():
    b = scale_boxes(np.array([[-50.0, -5.0, 700.0, 700.0]], np.float32), 0.5, 0, 80, 972, 1296)
    assert b[0, 0] == 0 and b[0, 1] == 0 and b[0, 2] == 1296 and b[0, 3] == 972


# ----------------------------- NMS e formatos ------------------------------ #
def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7], np.float32), 0.5)
    assert keep.tolist() == [0, 2]


@pytest.mark.parametrize("shape, nc, hint, layout", [
    ((1, 16, 8400), 12, False, "channels_first"),
    ((1, 8400, 16), 12, False, "channels_last"),
    ((1, 300, 6), 12, True, "end2end"),
    ((1, 300, 6), None, None, "end2end"),
    ((1, 8400, 6), 2, False, "channels_last"),
    ((1, 6, 8400), 2, False, "channels_first"),
])
def test_detect_layout(shape, nc, hint, layout):
    assert detect_layout(shape, nc, hint) == layout


def test_detect_layout_rejects_mismatch():
    with pytest.raises(ModelError):
        detect_layout((1, 20, 8400), 12, False)
    with pytest.raises(ModelError):
        detect_layout((8400, 16), 12, False)


def _classic_output(nc=3):
    out = np.zeros((1, 4 + nc, 100), np.float32)
    # duas caixas sobrepostas da classe 1 e uma isolada da classe 2
    for i, (cx, cy, s, c) in enumerate([(100, 100, 0.9, 1), (102, 101, 0.8, 1), (400, 400, 0.6, 2)]):
        out[0, :4, i] = (cx, cy, 50, 40)
        out[0, 4 + c, i] = s
    return out


def test_postprocess_classic_and_transposed_agree():
    out = _classic_output()
    b1, s1, c1 = postprocess(out, "channels_first", 0.25, 0.7, 300)
    b2, s2, c2 = postprocess(np.ascontiguousarray(out.transpose(0, 2, 1)), "channels_last", 0.25, 0.7, 300)
    assert c1.tolist() == c2.tolist() == [1, 2]
    np.testing.assert_allclose(s1, [0.9, 0.6])
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(b1[0], [75, 80, 125, 120])


def test_postprocess_end2end():
    out = np.array([[[10, 20, 30, 40, 0.9, 3], [0, 0, 5, 5, 0.1, 1], [50, 50, 70, 80, 0.5, 0]]], np.float32)
    b, s, c = postprocess(out, "end2end", 0.25, 0.7, 300)
    assert c.tolist() == [3, 0]
    np.testing.assert_allclose(b[0], [10, 20, 30, 40])


def test_postprocess_empty():
    b, s, c = postprocess(np.zeros((1, 16, 8400), np.float32), "channels_first", 0.25, 0.7, 300)
    assert b.shape == (0, 4) and s.size == 0 and c.size == 0


def test_parse_class_names():
    assert parse_class_names({"names": "{0: 'a', 1: 'b'}"}) == ["a", "b"]
    assert parse_class_names({"names": "lixo("}) is None
    assert parse_class_names({}) is None


# ----------------------------- modelo real -------------------------------- #
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="models/teste2.onnx ausente")


@pytest.fixture(scope="module")
def detector():
    return Detector(MODEL, intra_op_num_threads=2)


@needs_model
def test_real_model_io_verified_at_runtime(detector):
    assert detector.imgsz == 640
    assert detector.output_shape == (1, 16, 8400)
    assert detector.layout == "channels_first"
    assert detector.class_names == EXPECTED_CLASSES
    assert detector.metadata.get("end2end") == "False"


@needs_model
def test_real_model_blank_image(detector):
    res = detector.detect(np.full((972, 1296, 3), 114, np.uint8))
    assert res.detections == []
    assert res.inference_ms > 0


def _server_cases():
    if not SERVER_DATA.exists():
        return []
    cases = []
    for js in sorted(SERVER_DATA.glob("*/*/*.json")):
        jpg = js.with_suffix(".jpg")
        if jpg.exists():
            cases.append(pytest.param(jpg, js, id=f"{js.parent.name}/{js.stem[:8]}"))
    return cases


@needs_model
@pytest.mark.parametrize("jpg, js", _server_cases())
def test_matches_server_saved_results(detector, jpg, js):
    """Mesmas classes, mesma quantidade, confiança ±0,01 e caixas ±2 px do servidor."""
    expected = json.loads(js.read_text(encoding="utf-8"))["detections"]
    got = [d.to_wire() for d in detector.detect_jpeg(jpg.read_bytes()).detections]
    key = lambda d: (-d["confidence"], d["class_name"])  # noqa: E731
    expected, got = sorted(expected, key=key), sorted(got, key=key)
    assert [d["class_name"] for d in got] == [d["class_name"] for d in expected]
    for g, e in zip(got, expected):
        assert g["confidence"] == pytest.approx(e["confidence"], abs=0.01)
        np.testing.assert_allclose(g["bbox"], e["bbox"], atol=2.0)


@needs_model
def test_model_options_imgsz_and_class_names():
    names = [f"c{i}" for i in range(12)]
    det = Detector(MODEL, class_names=names, imgsz=640)
    assert det.class_names == names
    with pytest.raises(ModelError, match="imgsz"):
        Detector(MODEL, imgsz=320)


def test_resolve_imgsz_dynamic_input():
    assert Detector._resolve_imgsz([1, 3, "h", "w"], 416) == 416
    with pytest.raises(ModelError, match="dinâmica"):
        Detector._resolve_imgsz([1, 3, "h", "w"], 0)
