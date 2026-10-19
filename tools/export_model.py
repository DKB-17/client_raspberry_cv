"""Exporta o modelo treinado (.pt) para ONNX. RODA NO PC DE DESENVOLVIMENTO, não no Pi.

Requer Ultralytics (e PyTorch) instalados no PC:
    pip install ultralytics
    python tools/export_model.py caminho/para/best.pt
    python tools/export_model.py best.pt --end2end      # tenta saída [1,N,6] sem NMS

Parâmetros iguais aos do modelo atual do servidor (metadados de teste2.onnx):
imgsz=640, simplify=True, batch=1, dynamic=False, nms=False, FP32.
O nó detecta automaticamente o formato da saída ([1,4+nc,8400] ou [1,N,6]).
Depois copie o .onnx para client_raspberry_cv/models/ e ajuste model.path no config.json.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("weights", type=Path, help="arquivo .pt treinado (YOLO26n)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=None, help="opset ONNX (padrão do Ultralytics)")
    p.add_argument("--end2end", action="store_true", help="exporta a cabeça end-to-end (sem NMS)")
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "models")
    args = p.parse_args()

    if not args.weights.is_file():
        print(f"ERRO: {args.weights} não encontrado", file=sys.stderr)
        return 2
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: instale o Ultralytics no PC: pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    kwargs = dict(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False, batch=1, half=False)
    if args.opset:
        kwargs["opset"] = args.opset
    if args.end2end:
        kwargs["end2end"] = True  # suportado em versões recentes do Ultralytics para YOLO26
    exported = Path(model.export(**kwargs))

    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / exported.name
    if exported.resolve() != target.resolve():
        shutil.copy2(exported, target)
    print(f"ONNX gerado: {target}")
    _describe(target)
    return 0


def _describe(path: Path) -> None:
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


if __name__ == "__main__":
    sys.exit(main())
