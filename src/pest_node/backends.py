"""Adapters de runtime do modelo YOLO (padrão adapter).

O Detector só conhece a interface `ModelBackend`; cada runtime é um adapter:
  - "onnx"   -> OnnxBackend   (ONNX Runtime, arquivo .onnx, entrada NCHW)
  - "litert" -> LiteRTBackend (LiteRT/TFLite, arquivo .tflite exportado pelo
                Ultralytics com format="tflite", entrada NHWC)

A tag do runtime e o nome do arquivo vêm do .env (PEST_MODEL_BACKEND e
PEST_MODEL_NAME, ver .env.example). Para um runtime novo, crie uma classe com a
mesma interface e registre-a em BACKENDS.
"""

from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

InputLayout = Literal["nchw", "nhwc"]


class ModelError(RuntimeError):
    pass


class ModelBackend(Protocol):
    name: str  # tag do .env ("onnx" ou "litert")
    imgsz: int
    input_layout: InputLayout
    input_name: str
    output_name: str
    output_shape: tuple[Any, ...]
    metadata: dict[str, str]  # metadados do Ultralytics como texto (names, end2end, imgsz...)
    # True: caixas em 0..1 (multiplicar por imgsz); False: já em pixels; None: decide pela saída.
    normalized_boxes: bool | None
    threads: int

    def run(self, x: np.ndarray) -> np.ndarray: ...


def resolve_imgsz(h: object, w: object, imgsz: int, kind: str = "ONNX") -> int:
    """Tamanho da entrada: o do modelo (estático) ou o configurado (entrada dinâmica)."""
    if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
        if h != w:
            raise ModelError(f"entrada não quadrada: {h}x{w}")
        if imgsz and imgsz != h:
            raise ModelError(f"model.imgsz={imgsz}, mas o {kind} foi exportado com {h}x{w}; "
                             "use 0 (automático) ou exporte de novo")
        return h
    if not imgsz:
        raise ModelError(f"o {kind} tem entrada dinâmica; defina model.imgsz (ex.: 640)")
    return imgsz


# --------------------------------------------------------------------------- #
#  ONNX Runtime
# --------------------------------------------------------------------------- #
class OnnxBackend:
    name = "onnx"
    input_layout: InputLayout = "nchw"
    normalized_boxes = False

    def __init__(self, model_path: Path, intra_op_num_threads: int = 2, inter_op_num_threads: int = 1,
                 imgsz: int = 0) -> None:
        import onnxruntime as ort

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
        if len(inp.shape) != 4:
            raise ModelError(f"entrada com forma não suportada: {inp.shape}")
        if inp.type != "tensor(float)":
            raise ModelError(f"entrada deve ser float32, é {inp.type}")
        self.input_shape = list(inp.shape)
        self.imgsz = resolve_imgsz(inp.shape[2], inp.shape[3], imgsz, "ONNX")
        out = self.session.get_outputs()[0]
        self.output_name = out.name
        self.output_shape = tuple(out.shape)
        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.threads = intra_op_num_threads

    def run(self, x: np.ndarray) -> np.ndarray:
        return self.session.run([self.output_name], {self.input_name: x})[0]


# --------------------------------------------------------------------------- #
#  LiteRT (antigo TensorFlow Lite)
# --------------------------------------------------------------------------- #
def _litert_interpreter_class() -> type:
    """Interpreter do pacote ai-edge-litert; tflite_runtime como alternativa."""
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[no-redef]
        return Interpreter
    except ImportError:
        pass
    raise ModelError("runtime LiteRT não instalado; rode: pip install -r requirements.txt (pacote ai-edge-litert)")


def read_tflite_metadata(model_path: Path) -> dict[str, str]:
    """Metadados que o Ultralytics anexa ao .tflite (um zip com metadata.txt/metadata.json).

    Converte para o mesmo formato texto do custom_metadata_map do ONNX.
    """
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            name = zf.namelist()[0]
            contents = zf.read(name).decode("utf-8")
    except (zipfile.BadZipFile, IndexError, OSError, UnicodeDecodeError):
        return {}
    try:
        meta = json.loads(contents) if name.endswith(".json") else ast.literal_eval(contents)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(meta, dict):
        return {}
    names = meta.get("names")
    if isinstance(names, dict):  # chaves viram texto no JSON: "10" < "2" na ordenação
        meta["names"] = {int(k): str(v) for k, v in names.items()}
    return {str(k): str(v) for k, v in meta.items()}


def _quantize(x: np.ndarray, dtype: np.dtype, scale: float, zero_point: int) -> np.ndarray:
    info = np.iinfo(dtype)
    return np.clip(np.round(x / scale + zero_point), info.min, info.max).astype(dtype)


def _dequantize(q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (q.astype(np.float32) - zero_point) * np.float32(scale)


class LiteRTBackend:
    name = "litert"
    # O export TFLite antigo do Ultralytics (onnx2tf, entrada NHWC) devolve caixas em 0..1;
    # o novo (8.4+, entrada NCHW) já devolve em pixels. Decide pela saída a cada quadro.
    normalized_boxes = None

    def __init__(self, model_path: Path, intra_op_num_threads: int = 2, inter_op_num_threads: int = 1,
                 imgsz: int = 0) -> None:
        interpreter_cls = _litert_interpreter_class()
        try:
            self.interpreter = interpreter_cls(model_path=str(model_path), num_threads=intra_op_num_threads)
        except Exception as exc:
            raise ModelError(f"falha ao carregar {model_path}: {exc}") from exc

        inp = self.interpreter.get_input_details()[0]
        shape = [int(d) for d in inp.get("shape_signature", inp["shape"])]
        if len(shape) != 4:
            raise ModelError(f"entrada com forma não suportada: {shape}")
        # Ultralytics exporta NHWC [1, H, W, 3]; aceita também NCHW [1, 3, H, W].
        self.input_layout: InputLayout = "nchw" if shape[1] == 3 and shape[3] != 3 else "nhwc"
        h, w = (shape[2], shape[3]) if self.input_layout == "nchw" else (shape[1], shape[2])
        self.imgsz = resolve_imgsz(h, w, imgsz, "TFLite")
        if h <= 0 or w <= 0:  # entrada dinâmica: fixa no imgsz configurado
            full = ([1, 3, self.imgsz, self.imgsz] if self.input_layout == "nchw"
                    else [1, self.imgsz, self.imgsz, 3])
            self.interpreter.resize_tensor_input(inp["index"], full)
        try:
            self.interpreter.allocate_tensors()
        except Exception as exc:
            raise ModelError(f"falha ao alocar tensores de {model_path}: {exc}") from exc

        inp = self.interpreter.get_input_details()[0]
        out = self.interpreter.get_output_details()[0]
        self.input_name, self.output_name = inp["name"], out["name"]
        self.input_shape = [int(d) for d in inp["shape"]]
        self.output_shape = tuple(int(d) for d in out["shape"])
        self._in_index, self._out_index = inp["index"], out["index"]
        self._in_dtype = np.dtype(inp["dtype"])
        self._in_quant = tuple(inp.get("quantization", (0.0, 0)))
        self._out_quant = tuple(out.get("quantization", (0.0, 0)))
        self._out_integer = np.issubdtype(np.dtype(out["dtype"]), np.integer)
        if not (self._in_dtype == np.float32 or self._in_dtype == np.float16
                or np.issubdtype(self._in_dtype, np.integer)):
            raise ModelError(f"tipo de entrada não suportado: {self._in_dtype}")
        self.metadata = read_tflite_metadata(model_path)
        self.threads = intra_op_num_threads

    def run(self, x: np.ndarray) -> np.ndarray:
        if np.issubdtype(self._in_dtype, np.integer):
            scale, zp = self._in_quant
            x = _quantize(x, self._in_dtype, scale, int(zp))
        elif x.dtype != self._in_dtype:
            x = x.astype(self._in_dtype)
        self.interpreter.set_tensor(self._in_index, x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self._out_index)
        if self._out_integer:
            scale, zp = self._out_quant
            return _dequantize(y, scale, int(zp))
        return y.astype(np.float32, copy=True)


# --------------------------------------------------------------------------- #
#  Fábrica
# --------------------------------------------------------------------------- #
BACKENDS: dict[str, type] = {"onnx": OnnxBackend, "litert": LiteRTBackend}
MODEL_EXTENSIONS = {"onnx": ".onnx", "litert": ".tflite"}


def create_backend(kind: str, model_path: Path, intra_op_num_threads: int = 2, inter_op_num_threads: int = 1,
                   imgsz: int = 0) -> ModelBackend:
    """Instancia o adapter pela tag ("onnx" | "litert")."""
    cls = BACKENDS.get(kind.strip().lower())
    if cls is None:
        raise ModelError(f"backend de modelo desconhecido: {kind!r} (use: {', '.join(BACKENDS)})")
    if not Path(model_path).is_file():
        raise ModelError(f"modelo não encontrado: {model_path}")
    return cls(Path(model_path), intra_op_num_threads, inter_op_num_threads, imgsz)
