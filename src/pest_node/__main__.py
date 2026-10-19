"""CLI do nó sensor.

    pest-node run                      inicia o nó (usado pelo systemd)
    pest-node scan-qr [--timeout 120]  lê o QR do servidor e grava no config.json
    pest-node set-server URL           grava a URL manualmente (sem QR)
    pest-node selftest                 checa câmera, modelo, rede, servidor e sensores
    pest-node benchmark [...]          mede FPS/latência/CPU/RAM/temperatura -> CSV
    pest-node show-config              imprime a configuração efetiva
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from . import __version__
from . import config as cfgmod
from .config import ConfigError
from .logging_setup import get_logger, log_event, setup_logging

log = get_logger("main")


def _setup(cfg: cfgmod.Config) -> None:
    lg = cfg.logging
    setup_logging(lg.level, cfg.resolve(lg.dir), lg.max_bytes, lg.backup_count)


def cmd_run(cfg: cfgmod.Config, _args: argparse.Namespace) -> int:
    from .runtime import Node

    _setup(cfg)
    log_event(log, "starting", version=__version__, pid=os.getpid())
    node = Node(cfg)

    async def main() -> None:
        loop = asyncio.get_running_loop()
        if os.name == "posix":
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, node.stop_event.set)
        await node.run()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    return 0


def cmd_scan_qr(cfg: cfgmod.Config, args: argparse.Namespace) -> int:
    from .camera import make_camera
    from .discovery import scan_for_server_url
    from .selftest import _ws_ping

    _setup(cfg)
    c = cfg.camera
    cam = make_camera(c.width, c.height, c.max_fps, cfg.resolve(c.fake_images_dir) if c.fake_images_dir else None)
    cam.start()
    print(f"Aponte a câmera para o QR Code do servidor (até {args.timeout:.0f} s)…")
    try:
        url = scan_for_server_url(cam.capture, timeout_s=args.timeout)
    except TimeoutError as exc:
        print(f"ERRO: {exc}. Dicas: QR com 10 cm ou mais, a ~1 m da lente (foco fixo), boa luz.")
        return 2
    finally:
        cam.stop()
    return _save_server(cfg, url, check=not args.no_ping, ping=_ws_ping)


def cmd_set_server(cfg: cfgmod.Config, args: argparse.Namespace) -> int:
    from .discovery import InvalidServerUrl, normalize_server_url
    from .selftest import _ws_ping

    try:
        url = normalize_server_url(args.url)
    except InvalidServerUrl as exc:
        print(f"ERRO: URL inválida: {exc}")
        return 2
    return _save_server(cfg, url, check=not args.no_ping, ping=_ws_ping)


def _save_server(cfg: cfgmod.Config, url: str, check: bool, ping) -> int:
    if check:
        try:
            asyncio.run(ping(url))
            print(f"Servidor respondeu ao ping em {url}")
        except Exception as exc:
            print(f"AVISO: servidor não respondeu agora ({type(exc).__name__}: {exc}). "
                  "A URL será gravada mesmo assim; os alertas ficam na fila até ele voltar.")
    cfg.server_ws_url = url
    path = cfgmod.save(cfg)
    print(f"Gravado em {path}: server_ws_url = {url}")
    print("Reinicie o serviço para aplicar: sudo systemctl restart pest-node")
    return 0


def cmd_selftest(cfg: cfgmod.Config, args: argparse.Namespace) -> int:
    from .selftest import run_selftest

    setup_logging("WARNING")
    return run_selftest(cfg, args.config_path)


def cmd_show_config(cfg: cfgmod.Config, _args: argparse.Namespace) -> int:
    print(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pest-node", description="Nó sensor de detecção de pragas")
    p.add_argument("--config", type=Path, default=None,
                   help=f"caminho do config.json (padrão: ${cfgmod.CONFIG_ENV_VAR} ou ./config.json)")
    p.add_argument("--version", action="version", version=f"pest-node {__version__}")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="inicia o nó")
    s = sub.add_parser("scan-qr", help="força nova leitura do QR Code do servidor")
    s.add_argument("--timeout", type=float, default=120.0)
    s.add_argument("--no-ping", action="store_true", help="não testa o servidor antes de gravar")
    s = sub.add_parser("set-server", help="grava a URL do servidor manualmente")
    s.add_argument("url")
    s.add_argument("--no-ping", action="store_true")
    sub.add_parser("selftest", help="autoteste com relatório")
    sub.add_parser("show-config", help="mostra a configuração efetiva")
    sub.add_parser("benchmark", help="benchmark de carga contínua (use: benchmark --help)", add_help=False)
    return p


COMMANDS = {"run": cmd_run, "scan-qr": cmd_scan_qr, "set-server": cmd_set_server,
            "selftest": cmd_selftest, "show-config": cmd_show_config}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    if "benchmark" in argv:  # repassa as opções próprias do benchmark
        from .bench import main as bench_main

        i = argv.index("benchmark")
        rest = argv[i + 1:]
        if "--config" in argv[:i]:
            j = argv.index("--config")
            rest = ["--config", argv[j + 1], *rest]
        try:
            return bench_main(rest)
        except ConfigError as exc:
            print(f"ERRO de configuração: {exc}", file=sys.stderr)
            return 2
    args = parser.parse_args(argv)
    path = (args.config or cfgmod.default_path()).expanduser().resolve()
    args.config_path = path
    try:
        cfg = cfgmod.load(path)
    except ConfigError as exc:
        print(f"ERRO de configuração: {exc}", file=sys.stderr)
        return 2
    return COMMANDS[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
