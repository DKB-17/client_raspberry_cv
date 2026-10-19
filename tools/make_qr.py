"""Gera um PNG com o QR Code da URL do servidor, para imprimir ou mostrar na tela.

O conteúdo é o mesmo que o servidor_cv imprime no terminal:
"ws://<ip>:<porta>/ws?capture_interval=<segundos>". Usa apenas o OpenCV.

    python tools/make_qr.py ws://192.168.5.12:8000/ws --interval 5
    python tools/make_qr.py "ws://192.168.5.12:8000/ws?capture_interval=5" --out qr_servidor.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pest_node.discovery import CAPTURE_INTERVAL_PARAM, InvalidServerUrl, decode_qr, parse_server_qr  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("--out", type=Path, default=Path("qr_servidor.png"))
    p.add_argument("--size", type=int, default=1000, help="lado da imagem em pixels")
    p.add_argument("--interval", type=float, default=None, help="intervalo de captura em segundos")
    args = p.parse_args()
    try:
        info = parse_server_qr(args.url)
        interval = args.interval if args.interval is not None else info.capture_interval_s
        url = info.url if interval is None else f"{info.url}?{CAPTURE_INTERVAL_PARAM}={interval:g}"
        parse_server_qr(url)  # valida a faixa do intervalo
    except InvalidServerUrl as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2
    qr = cv2.QRCodeEncoder.create().encode(url)
    border = max(4, qr.shape[0] // 6)
    qr = cv2.copyMakeBorder(qr, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(args.out), qr)
    check = decode_qr(cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR))
    print(f"QR gravado em {args.out} com o conteúdo: {url} (releitura: {'ok' if check == url else 'FALHOU'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
