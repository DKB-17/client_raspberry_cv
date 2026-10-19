"""Testes dos adapters de runtime (backends.py).

O LiteRT é testado com um interpreter falso (não há .tflite no repositório),
que imita a API do ai_edge_litert.interpreter.Interpreter.
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pest_node import backends
from pest_node.backends import (LiteRTBackend, ModelError, OnnxBackend, _dequantize, _quantize, create_backend,
                                read_tflite_metadata)
from pest_node.detector import Detector, boxes_are_normalized, denormalize_boxes

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "teste2.onnx"
NAMES = [f"praga_{i}" for i in range(12)]


def _fake_tflite(path: Path, meta_name: str = "metadata.txt", meta: str | None = None) -> Path:
    """Arquivo .tflite falso com metadados anexados como o Ultralytics faz (zip no fim)."""
    path.write_bytes(b"TFL3" + b"\0" * 64)
    if meta is None:
        meta = repr({"names": dict(enumerate(NAMES)), "imgsz": [640, 640], "end2end": False, "task": "detect"})
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(meta_name, meta)
    return path


def _make_interpreter(in_dtype=np.float32, out_dtype=np.float32, in_quant=(0.0, 0), out_quant=(0.0, 0),
                      in_shape=(1, 640, 640, 3)):
    class FakeInterpreter:
        received: list[np.ndarray] = []

        def __init__(self, model_path=None, num_threads=None):
            self.model_path, self.num_threads = model_path, num_threads
            self.in_shape = list(in_shape)

        def get_input_details(self):
            return [{"name": "images", "index": 0, "shape": np.array(self.in_shape),
                     "shape_signature": np.array(in_shape), "dtype": in_dtype, "quantization": in_quant}]

        def get_output_details(self):
            return [{"name": "output0", "index": 1, "shape": np.array([1, 16, 8400]), "dtype": out_dtype,
                     "quantization": out_quant}]

        def resize_tensor_input(self, index, shape):
            self.in_shape = list(shape)

        def allocate_tensors(self):
            pass

        def set_tensor(self, index, value):
            FakeInterpreter.received.append(value)

        def invoke(self):
            pass

        def get_tensor(self, index):
            # Uma caixa normalizada no centro: cx=cy=0.5, w=h=0.25, classe 5 com 0.9.
            out = np.zeros((1, 16, 8400), np.float32)
            out[0, :4, 0] = [0.5, 0.5, 0.25, 0.25]
            out[0, 4 + 5, 0] = 0.9
            if np.issubdtype(np.dtype(out_dtype), np.integer):
                scale, zp = out_quant
                return np.round(out / scale + zp).astype(out_dtype)
            return out.astype(out_dtype)

    return FakeInterpreter


@pytest.fixture
def fake_litert(monkeypatch):
    def install(**kwargs):
        cls = _make_interpreter(**kwargs)
        monkeypatch.setattr(backends, "_litert_interpreter_class", lambda: cls)
        return cls
    return install


# ------------------------------- fábrica ---------------------------------- #
def test_factory_picks_onnx_adapter():
    b = create_backend("onnx", MODEL)
    assert isinstance(b, OnnxBackend)
    assert b.input_layout == "nchw" and b.imgsz == 640 and not b.normalized_boxes


def test_factory_rejects_unknown_tag_and_missing_file(tmp_path):
    with pytest.raises(ModelError, match="desconhecido"):
        create_backend("tensorrt", MODEL)
    with pytest.raises(ModelError, match="não encontrado"):
        create_backend("litert", tmp_path / "nao_existe.tflite")


def test_litert_missing_runtime_gives_clear_error(tmp_path, monkeypatch):
    for mod in ("ai_edge_litert", "ai_edge_litert.interpreter", "tflite_runtime", "tflite_runtime.interpreter"):
        monkeypatch.setitem(sys.modules, mod, None)
    with pytest.raises(ModelError, match="ai-edge-litert"):
        create_backend("litert", _fake_tflite(tmp_path / "m.tflite"))


# ------------------------------- LiteRT ----------------------------------- #
def test_litert_adapter_reads_shape_threads_and_metadata(tmp_path, fake_litert):
    fake_litert()
    b = create_backend("litert", _fake_tflite(tmp_path / "m.tflite"), intra_op_num_threads=3)
    assert isinstance(b, LiteRTBackend)
    assert b.input_layout == "nhwc" and b.imgsz == 640 and b.normalized_boxes is None
    assert b.interpreter.num_threads == 3 and b.threads == 3
    assert b.output_shape == (1, 16, 8400)
    assert b.metadata["end2end"] == "False"


def test_litert_detector_end_to_end(tmp_path, fake_litert):
    cls = fake_litert()
    det = Detector(_fake_tflite(tmp_path / "m.tflite"), backend="litert")
    assert det.class_names == NAMES
    assert det.layout == "channels_first"
    image = np.zeros((640, 640, 3), np.uint8)
    image[..., 0] = 255  # azul em BGR -> deve chegar como último canal (RGB)
    res = det.detect(image)
    sent = cls.received[-1]
    assert sent.shape == (1, 640, 640, 3) and sent.dtype == np.float32
    assert sent[0, 0, 0, 2] == pytest.approx(1.0) and sent[0, 0, 0, 0] == 0.0
    assert len(res.detections) == 1
    d = res.detections[0]
    assert d.class_name == "praga_5" and d.confidence == pytest.approx(0.9)
    assert d.bbox == pytest.approx((240.0, 240.0, 400.0, 400.0))
    assert det.describe()["backend"] == "litert"


def test_litert_int8_quantized_io(tmp_path, fake_litert):
    cls = fake_litert(in_dtype=np.int8, out_dtype=np.int8, in_quant=(1 / 255, -128), out_quant=(0.01, 0))
    det = Detector(_fake_tflite(tmp_path / "m.tflite"), backend="litert")
    res = det.detect(np.full((640, 640, 3), 255, np.uint8))
    sent = cls.received[-1]
    assert sent.dtype == np.int8 and int(sent.max()) == 127
    assert res.detections[0].bbox == pytest.approx((240.0, 240.0, 400.0, 400.0), abs=1.0)


def test_litert_dynamic_input_uses_configured_imgsz(tmp_path, fake_litert):
    fake_litert(in_shape=(1, -1, -1, 3))
    with pytest.raises(ModelError, match="dinâmica"):
        create_backend("litert", _fake_tflite(tmp_path / "m.tflite"))
    b = create_backend("litert", tmp_path / "m.tflite", imgsz=320)
    assert b.imgsz == 320 and b.input_shape == [1, 320, 320, 3]


def test_tflite_metadata_json_keys_are_sorted_numerically(tmp_path):
    names = {str(i): f"c{i}" for i in range(12)}
    path = _fake_tflite(tmp_path / "m.tflite", "metadata.json",
                        '{"names": %s, "end2end": true}' % str(names).replace("'", '"'))
    meta = read_tflite_metadata(path)
    from pest_node.detector import parse_class_names

    assert parse_class_names(meta) == [f"c{i}" for i in range(12)]
    assert meta["end2end"] == "True"


def test_tflite_without_metadata(tmp_path):
    path = tmp_path / "m.tflite"
    path.write_bytes(b"TFL3 sem zip")
    assert read_tflite_metadata(path) == {}


# ------------------------------- auxiliares -------------------------------- #
def test_quantize_roundtrip():
    x = np.array([0.0, 0.2, 1.0, 2.0], np.float32)
    q = _quantize(x, np.dtype(np.int8), 1 / 255, -128)
    assert q.tolist() == [-128, -77, 127, 127]  # satura no limite do int8
    x[3] = 1.0
    assert _dequantize(q, 1 / 255, -128) == pytest.approx(x, abs=1 / 255)


@pytest.mark.parametrize("layout, value, expected", [
    ("channels_first", 0.5, True), ("channels_first", 320.0, False),
    ("end2end", 0.9, True), ("end2end", 500.0, False),
])
def test_boxes_are_normalized_auto(layout, value, expected):
    shape = (1, 6, 3) if layout == "channels_first" else (1, 3, 6)
    assert boxes_are_normalized(np.full(shape, value, np.float32), layout) is expected
    assert boxes_are_normalized(np.full(shape, value, np.float32), layout, hint=False) is False


def test_litert_pixel_boxes_are_not_rescaled(tmp_path, monkeypatch, fake_litert):
    """Export novo do Ultralytics (8.4+): entrada NCHW e caixas já em pixels."""
    cls = fake_litert(in_shape=(1, 3, 640, 640))
    pixel = cls.get_tensor

    def get_tensor(self, index):
        out = pixel(self, index)
        out[0, :4, 0] *= 640
        return out
    monkeypatch.setattr(cls, "get_tensor", get_tensor)
    det = Detector(_fake_tflite(tmp_path / "m.tflite"), backend="litert")
    res = det.detect(np.zeros((640, 640, 3), np.uint8))
    assert cls.received[-1].shape == (1, 3, 640, 640)
    assert res.detections[0].bbox == pytest.approx((240.0, 240.0, 400.0, 400.0))


@pytest.mark.parametrize("layout, shape", [("channels_first", (1, 6, 3)), ("channels_last", (1, 3, 6)),
                                           ("end2end", (1, 3, 6))])
def test_denormalize_boxes_scales_only_coordinates(layout, shape):
    out = np.full(shape, 0.5, np.float32)
    denormalize_boxes(out, layout, 640)
    boxes = out[0, :4, :] if layout == "channels_first" else out[0, :, :4]
    rest = out[0, 4:, :] if layout == "channels_first" else out[0, :, 4:]
    assert np.all(boxes == 320) and np.all(rest == 0.5)
