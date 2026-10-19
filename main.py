"""Nó sensor de pragas: ponto de entrada único.

Fluxo de execução (python main.py):

  1. Configuração  carrega config.json (modelo, câmera, intervalo, limiares...).
  2. Serviços      carrega o modelo YOLO (ONNX) e sobe a fila de alertas, o
                   WebSocket do app e a leitura de temperatura.
  3. Câmera        abre o módulo de câmera (tenta de novo se falhar).
  4. Servidor      se não houver servidor configurado (ou com --scan-qr), lê o
                   QR Code do servidor: ele traz a URL e o intervalo de captura.
  5. Coleta        a cada `inference.capture_interval_s` segundos: captura um
                   quadro -> analisa com o YOLO -> se houver praga acima do
                   limiar, envia a imagem ao servidor (alerta).

Outros usos:
  python main.py --scan-qr            força nova leitura do QR e segue rodando
  python main.py --set-server URL     grava o servidor sem QR e sai
  python main.py --selftest           relatório de câmera, modelo, rede e sensores
  python main.py --show-config        mostra a configuração efetiva
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pest_node import __version__  # noqa: E402
from pest_node import config as cfgmod  # noqa: E402
from pest_node.camera import Camera  # noqa: E402
from pest_node.config import Config, ConfigError  # noqa: E402
from pest_node.discovery import InvalidServerUrl, parse_server_qr, scan_for_server_url  # noqa: E402
from pest_node.logging_setup import get_logger, log_event, setup_logging  # noqa: E402
from pest_node.runtime import Node  # noqa: E402
from pest_node.thermal import ThermalSensor  # noqa: E402

log = get_logger("main")

QR_TIMEOUT_S = 120.0  # tempo de leitura do QR na inicialização; depois segue e continua procurando
CAMERA_RETRY_S = 30.0


# --------------------------------------------------------------------------- #
#  Etapas do fluxo
# --------------------------------------------------------------------------- #
def carregar_configuracao(path: Path | None) -> Config:
    """Lê e valida o config.json (erros de valor param aqui, com mensagem clara)."""
    return cfgmod.load(path)


def iniciar_logs(cfg: Config) -> None:
    lg = cfg.logging
    setup_logging(lg.level, cfg.resolve(lg.dir), lg.max_bytes, lg.backup_count)
    m, i = cfg.model, cfg.inference
    log_event(log, "config_loaded", version=__version__, node_id=cfg.node_id, model=m.path,
              conf=m.conf_threshold, iou=m.iou_threshold, threads=m.intra_op_num_threads,
              capture_interval_s=i.capture_interval_s, mode=i.mode, alert_threshold=cfg.alerts.threshold,
              server=cfg.server_ws_url or None)


async def abrir_camera(node: Node) -> None:
    while not await node.open_camera():
        log_event(log, "camera_retry", level=30, retry_in_s=CAMERA_RETRY_S)
        await asyncio.sleep(CAMERA_RETRY_S)  # enquanto isso o WebSocket do app segue atendendo


async def descobrir_servidor(node: Node, timeout_s: float = QR_TIMEOUT_S) -> None:
    log_event(log, "qr_scanning", timeout_s=timeout_s, hint="mostre o QR do servidor à câmera (~1 m)")
    try:
        info = await asyncio.to_thread(scan_for_server_url, node.camera.capture, timeout_s)
    except TimeoutError:
        log_event(log, "qr_not_found", level=30,
                  hint="seguindo sem servidor; o QR continua sendo procurado a cada quadro")
        return
    await node.use_server(info)


async def fluxo_de_coleta(node: Node) -> None:
    """Captura e analisa um quadro a cada capture_interval_s (ampliado se o Pi esquentar)."""
    while not node.stop_event.is_set():
        inicio = time.monotonic()
        try:
            await node.collect_once()
        except Exception:
            log.exception("erro no ciclo de coleta; tentando de novo no próximo intervalo")
        espera = max(0.0, node.interval_s - (time.monotonic() - inicio))
        try:
            await asyncio.wait_for(node.stop_event.wait(), espera)
        except asyncio.TimeoutError:
            pass


async def executar(cfg: Config, forcar_qr: bool = False, camera: Camera | None = None,
                   sensor: ThermalSensor | None = None, on_ready: Callable[[Node], None] | None = None) -> None:
    node = Node(cfg, camera=camera, sensor=sensor)
    if os.name == "posix":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, node.stop_event.set)
    await node.start()
    if on_ready:
        on_ready(node)
    try:
        await abrir_camera(node)
        if forcar_qr or not cfg.server_ws_url:
            await descobrir_servidor(node)
        log_event(log, "collection_started", interval_s=node.interval_s, mode=cfg.inference.mode,
                  server=cfg.server_ws_url or None)
        await fluxo_de_coleta(node)
    finally:
        await node.stop()


# --------------------------------------------------------------------------- #
#  Comandos auxiliares
# --------------------------------------------------------------------------- #
def gravar_servidor(cfg: Config, texto: str) -> int:
    from pest_node.selftest import _ws_ping

    try:
        info = parse_server_qr(texto)
    except InvalidServerUrl as exc:
        print(f"ERRO: URL inválida: {exc}")
        return 2
    try:
        asyncio.run(_ws_ping(info.url))
        print(f"Servidor respondeu ao ping em {info.url}")
    except Exception as exc:
        print(f"AVISO: servidor não respondeu agora ({type(exc).__name__}). A URL será gravada mesmo assim.")
    cfg.server_ws_url = info.url
    if info.capture_interval_s is not None:
        cfg.inference.capture_interval_s = info.capture_interval_s
    path = cfgmod.save(cfg)
    print(f"Gravado em {path}: server_ws_url={info.url} capture_interval_s={cfg.inference.capture_interval_s:g}")
    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nó sensor de detecção de pragas",
                                epilog="Sem opções: inicia o fluxo de coleta.")
    p.add_argument("--config", type=Path, default=None,
                   help=f"config.json (padrão: ${cfgmod.CONFIG_ENV_VAR} ou ./config.json)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--scan-qr", action="store_true", help="força nova leitura do QR na inicialização")
    g.add_argument("--set-server", metavar="URL", help="grava o servidor (ex.: ws://IP:8000/ws?capture_interval=5)")
    g.add_argument("--selftest", action="store_true", help="autoteste com relatório")
    g.add_argument("--show-config", action="store_true", help="mostra a configuração efetiva")
    p.add_argument("--version", action="version", version=f"pest-node {__version__}")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    path = (args.config or cfgmod.default_path()).expanduser().resolve()
    try:
        cfg = carregar_configuracao(path)
    except ConfigError as exc:
        print(f"ERRO de configuração: {exc}", file=sys.stderr)
        return 2
    if args.show_config:
        print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.set_server:
        return gravar_servidor(cfg, args.set_server)
    if args.selftest:
        from pest_node.selftest import run_selftest

        setup_logging("WARNING")
        return run_selftest(cfg, path)
    iniciar_logs(cfg)
    try:
        asyncio.run(executar(cfg, forcar_qr=args.scan_qr))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
