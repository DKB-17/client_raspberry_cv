"""Exporta o modelo treinado (.pt) para ONNX ou LiteRT. RODA NO PC DE DESENVOLVIMENTO, não no Pi.

Requer Ultralytics (e PyTorch) instalados no PC:
    pip install ultralytics
    python tools/export_model.py caminho/para/best.pt                    # ONNX (padrão)
    python tools/export_model.py best.pt --end2end                       # tenta saída [1,N,6] sem NMS
    python tools/export_model.py best.pt --format litert                 # LiteRT float32 (.tflite)
    python tools/export_model.py best.pt --format litert --half          # LiteRT float16
    python tools/export_model.py best.pt --format litert --int8 --data dados.yaml  # LiteRT int8

Parâmetros iguais aos do modelo atual do servidor (metadados de teste2.onnx):
imgsz=640, simplify=True, batch=1, dynamic=False, nms=False, FP32.
O nó detecta automaticamente o formato da saída ([1,4+nc,8400] ou [1,N,6]).
Depois copie o arquivo gerado para client_raspberry_cv/models/ e ajuste o .env
(PEST_MODEL_BACKEND e PEST_MODEL_NAME); o script imprime as linhas prontas.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", type=Path, help="arquivo .pt treinado (YOLO26n)")
    p.add_argument("--format", choices=("onnx", "litert"), default="onnx",
                   help="onnx (ONNX Runtime) ou litert (.tflite para LiteRT)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=None, help="opset ONNX (padrão do Ultralytics)")
    p.add_argument("--end2end", action="store_true", help="exporta a cabeça end-to-end (sem NMS)")
    p.add_argument("--half", action="store_true", help="LiteRT float16")
    p.add_argument("--int8", action="store_true", help="LiteRT int8 (requer --data para calibração)")
    p.add_argument("--data", default=None, help="dataset .yaml para calibração int8")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = p.parse_args()

    if not args.weights.is_file():
        print(f"ERRO: {args.weights} não encontrado", file=sys.stderr)
        return 2
    if args.format == "onnx" and (args.half or args.int8):
        print("ERRO: --half/--int8 valem só para --format litert", file=sys.stderr)
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: instale o Ultralytics no PC: pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    kwargs: dict[str, object] = dict(imgsz=args.imgsz, batch=1, dynamic=False)
    if args.format == "onnx":
        kwargs.update(format="onnx", simplify=True, half=False)
        if args.opset:
            kwargs["opset"] = args.opset
    else:
        kwargs.update(format="tflite", half=args.half, int8=args.int8)  # LiteRT = formato tflite
        if args.data:
            kwargs["data"] = args.data
    if args.end2end:
        kwargs["end2end"] = True  # suportado em versões recentes do Ultralytics para YOLO26
    exported = Path(model.export(**kwargs))

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / exported.name
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    print(f"Modelo gerado: {target}")
    if target.suffix == ".tflite":
        _describe_tflite(target)
    else:
        _describe_onnx(target)
    print("\nNo .env do Pi:")
    print(f"  PEST_MODEL_BACKEND={args.format}")
    print(f"  PEST_MODEL_NAME={target.name}")
    return 0


def _describe_onnx(path: Path) -> None:
    """Mostra a forma REAL dos tensores, para conferir antes de levar ao Pi."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("(instale onnxruntime para ver as formas dos tensores)")
        return
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for i in s.get_inputs():
        print(f"  entrada: {i.name} {i.shape} {i.type}")
    for o in s.get_outputs():
        print(f"  saída:   {o.name} {o.shape} {o.type}")
    meta = s.get_modelmeta().custom_metadata_map
    print(f"  end2end={meta.get('end2end')} imgsz={meta.get('imgsz')}")
    print(f"  classes={meta.get('names')}")


def _describe_tflite(path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("(instale ai-edge-litert para ver as formas dos tensores)")
        return
    from pest_node.backends import read_tflite_metadata

    it = Interpreter(model_path=str(path))
    for i in it.get_input_details():
        print(f"  entrada: {i['name']} {list(i['shape'])} {i['dtype'].__name__}")
    for o in it.get_output_details():
        print(f"  saída:   {o['name']} {list(o['shape'])} {o['dtype'].__name__}")
    meta = read_tflite_metadata(path)
    print(f"  end2end={meta.get('end2end')} imgsz={meta.get('imgsz')}")
    print(f"  classes={meta.get('names')}")


if __name__ == "__main__":
    sys.exit(main())
