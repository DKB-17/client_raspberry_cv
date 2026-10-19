"""Detector YOLO26n via ONNX Runtime (somente CPU).

Pré-processamento idêntico ao do Ultralytics (usado pelo servidor):
  letterbox centralizado para imgsz x imgsz com cinza (114,114,114),
  BGR -> RGB, HWC -> CHW, float32 / 255.

Pós-processamento escolhido automaticamente pela forma real da saída:
  - [1, 4+nc, A]  formato clássico (cx, cy, w, h + scores) -> NMS por classe
  - [1, A, 4+nc]  idem, transposto
  - [1, N, 6]     end-to-end (x1, y1, x2, y2, score, classe), sem NMS
O modelo do projeto (teste2.onnx) tem end2end=False e saída [1, 16, 8400].
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort

from .logging_setup import get_logger, log_event
from .models import Detection, InferenceResult

log = get_logger("detector")

OutputLayout = Literal["channels_first", "channels_last", "end2end"]
LETTERBOX_COLOR = (114, 114, 114)
_MAX_WH = 7680.0  # deslocamento por classe no NMS (mesmo valor do Ultralytics)
_MAX_NMS_CANDIDATES = 30000


class ModelError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
#  Letterbox e reescala (funções puras, testáveis)
# --------------------------------------------------------------------------- #
def letterbox_params(h: int, w: int, size: int) -> tuple[float, int, int, int, int]:
    """Retorna (ganho, nova_largura, nova_altura, pad_esq, pad_topo) como o Ultralytics."""
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    return r, new_w, new_h, left, top


def letterbox(image: np.ndarray, size: int, out: np.ndarray | None = None) -> tuple[np.ndarray, float, int, int]:
    """Redimensiona mantendo proporção e preenche com cinza. Reutiliza `out` se dado."""
    h, w = image.shape[:2]
    r, new_w, new_h, left, top = letterbox_params(h, w, size)
    if out is None:
        out = np.empty((size, size, 3), np.uint8)
    out[...] = LETTERBOX_COLOR
    region = out[top:top + new_h, left:left + new_w]
    if (new_w, new_h) == (w, h):
        region[...] = image
    else:
        cv2.resize(image, (new_w, new_h), dst=region, interpolation=cv2.INTER_LINEAR)
    return out, r, left, top


def scale_boxes(boxes: np.ndarray, gain: float, pad_x: int, pad_y: int, h: int, w: int) -> np.ndarray:
    """Converte xyxy do espaço do letterbox para pixels da imagem original (in-place)."""
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= gain
    boxes[:, 0] = boxes[:, 0].clip(0, w)
    boxes[:, 2] = boxes[:, 2].clip(0, w)
    boxes[:, 1] = boxes[:, 1].clip(0, h)
    boxes[:, 3] = boxes[:, 3].clip(0, h)
    return boxes


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh)
    half_w, half_h = xywh[:, 2] / 2, xywh[:, 3] / 2
    out[:, 0] = xywh[:, 0] - half_w
    out[:, 1] = xywh[:, 1] - half_h
    out[:, 2] = xywh[:, 0] + half_w
    out[:, 3] = xywh[:, 1] + half_h
    return out


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """NMS guloso em numpy. Retorna índices mantidos em ordem decrescente de score."""
    order = scores.argsort()[::-1]
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def detect_layout(shape: tuple[int, ...], num_classes: int | None,
                  end2end_hint: bool | None = None) -> OutputLayout:
    """Escolhe o pós-processamento pela forma real da saída do ONNX.

    `end2end_hint` vem dos metadados do Ultralytics e só desempata o caso
    ambíguo [1, N, 6] com 2 classes.
    """
    if len(shape) != 3 or shape[0] != 1:
        raise ModelError(f"saída com forma inesperada: {shape}")
    _, a, b = shape
    if b == 6 and a != 6:
        ambiguous = num_classes == 2
        if not ambiguous or end2end_hint is not False:
            return "end2end"
    if num_classes is not None:
        if a == 4 + num_classes:
            return "channels_first"
        if b == 4 + num_classes:
            return "channels_last"
        raise ModelError(f"saída {shape} incompatível com {num_classes} classes")
    return "channels_first" if a < b else "channels_last"


def postprocess(output: np.ndarray, layout: OutputLayout, conf_threshold: float, iou_threshold: float,
                max_det: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retorna (boxes_xyxy no espaço do letterbox, scores, class_ids)."""
    pred = output[0]
    if layout == "end2end":
        mask = pred[:, 4] > conf_threshold
        sel = pred[mask]
        sel = sel[sel[:, 4].argsort()[::-1][:max_det]]
        return sel[:, :4].astype(np.float32), sel[:, 4].astype(np.float32), sel[:, 5].astype(np.int64)
    if layout == "channels_first":
        pred = pred.T  # (A, 4+nc) — view, sem cópia
    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    mask = scores > conf_threshold
    if not mask.any():
        empty = np.empty((0,), np.float32)
        return np.empty((0, 4), np.float32), empty, np.empty((0,), np.int64)
    boxes = xywh_to_xyxy(pred[mask, :4].astype(np.float32))
    scores, class_ids = scores[mask].astype(np.float32), class_ids[mask]
    if scores.size > _MAX_NMS_CANDIDATES:
        top = scores.argsort()[::-1][:_MAX_NMS_CANDIDATES]
        boxes, scores, class_ids = boxes[top], scores[top], class_ids[top]
    keep = nms(boxes + (class_ids[:, None] * _MAX_WH), scores, iou_threshold)[:max_det]
    return boxes[keep], scores[keep], class_ids[keep]


def parse_class_names(meta: dict[str, str]) -> list[str] | None:
    raw = meta.get("names")
    if not raw:
        return None
    try:
        names = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names)]
    if isinstance(names, list):
        return [str(n) for n in names]
    return None


# --------------------------------------------------------------------------- #
#  Detector
# --------------------------------------------------------------------------- #
class Detector:
    def __init__(self, model_path: Path, conf_threshold: float = 0.25, iou_threshold: float = 0.7,
                 max_detections: int = 300, intra_op_num_threads: int = 2, inter_op_num_threads: int = 1,
                 class_names: list[str] | None = None) -> None:
        if not Path(model_path).is_file():
            raise ModelError(f"modelo não encontrado: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_num_threads
        opts.inter_op_num_threads = inter_op_num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=opts,
                                                providers=["CPUExecutionProvider"])
        except Exception as exc:  # onnxruntime lança tipos variados
            raise ModelError(f"falha ao carregar {model_path}: {exc}") from exc

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        if len(inp.shape) != 4 or not all(isinstance(d, int) for d in inp.shape[2:]):
            raise ModelError(f"entrada com forma não suportada: {inp.shape}")
        if inp.shape[2] != inp.shape[3]:
            raise ModelError(f"entrada não quadrada: {inp.shape}")
        if inp.type != "tensor(float)":
            raise ModelError(f"entrada deve ser float32, é {inp.type}")
        self.imgsz = int(inp.shape[2])
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        self.output_shape = tuple(out.shape)

        meta = self.session.get_modelmeta().custom_metadata_map
        self.metadata = dict(meta)
        self.class_names = class_names or parse_class_names(meta) or []
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        e2e = meta.get("end2end")
        self._end2end_hint = None if e2e is None else e2e.strip().lower() == "true"
        self.layout: OutputLayout | None = None
        if all(isinstance(d, int) for d in self.output_shape):
            self.layout = detect_layout(self.output_shape, len(self.class_names) or None, self._end2end_hint)

        # Buffers reutilizados a cada quadro (sem alocação por inferência).
        self._canvas = np.empty((self.imgsz, self.imgsz, 3), np.uint8)
        self._input = np.empty((1, 3, self.imgsz, self.imgsz), np.float32)

        log_event(log, "model_loaded", path=str(model_path), input=inp.shape, output=self.output_shape,
                  layout=self.layout, classes=len(self.class_names), end2end=meta.get("end2end"),
                  threads=intra_op_num_threads)

    def describe(self) -> dict[str, object]:
        return {
            "input": [self.input_name, [1, 3, self.imgsz, self.imgsz]],
            "output": [self.output_name, list(self.output_shape)],
            "layout": self.layout,
            "classes": self.class_names,
            "end2end": self.metadata.get("end2end"),
            "ultralytics_version": self.metadata.get("version"),
        }

    def preprocess(self, image_bgr: np.ndarray) -> tuple[float, int, int]:
        _, gain, pad_x, pad_y = letterbox(image_bgr, self.imgsz, out=self._canvas)
        chw_rgb = self._canvas.transpose(2, 0, 1)[::-1]  # view: CHW e BGR->RGB
        np.divide(chw_rgb, np.float32(255.0), out=self._input[0], dtype=np.float32)
        return gain, pad_x, pad_y

    def detect(self, image_bgr: np.ndarray) -> InferenceResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"imagem deve ser HxWx3 BGR, recebida {image_bgr.shape}")
        h, w = image_bgr.shape[:2]
        t0 = time.perf_counter()
        gain, pad_x, pad_y = self.preprocess(image_bgr)
        t1 = time.perf_counter()
        output = self.session.run([self.output_name], {self.input_name: self._input})[0]
        t2 = time.perf_counter()
        if self.layout is None:
            self.layout = detect_layout(output.shape, len(self.class_names) or None, self._end2end_hint)
        boxes, scores, class_ids = postprocess(output, self.layout, self.conf_threshold,
                                               self.iou_threshold, self.max_detections)
        boxes = scale_boxes(boxes, gain, pad_x, pad_y, h, w)
        detections = [
            Detection(int(c), self._name(int(c)), float(s), (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for b, s, c in zip(boxes, scores, class_ids)
        ]
        t3 = time.perf_counter()
        return InferenceResult(detections, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3)

    def detect_jpeg(self, jpeg: bytes) -> InferenceResult:
        image = decode_jpeg(jpeg)
        return self.detect(image)

    def _name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"classe_{class_id}"  # mesmo fallback do servidor


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Não foi possível decodificar a imagem recebida.")
    return image


def encode_jpeg(image_bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("falha ao codificar JPEG")
    return buf.tobytes()
