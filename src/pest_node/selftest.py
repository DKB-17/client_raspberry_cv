"""Autoteste: câmera, modelo, rede, servidor, sensores, disco e memória.

Imprime um relatório legível e retorna 0 se nada falhou (avisos não falham).
"""

from __future__ import annotations

import asyncio
import json
import platform
import shutil
import socket
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np

from . import __version__
from . import config as cfgmod
from .camera import make_camera
from .config import Config
from .detector import Detector
from .discovery import http_base_url
from .outbox import Outbox
from .sysmetrics import mem_available_mb
from .thermal import VcgencmdSensor

OK, WARN, FAIL = "OK", "AVISO", "FALHA"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def _run(name: str, fn: Callable[[], tuple[str, str]]) -> Check:
    try:
        status, detail = fn()
    except Exception as exc:  # o autoteste nunca deve quebrar
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    return Check(name, status, detail)


def check_platform() -> tuple[str, str]:
    arch, py = platform.machine(), platform.python_version()
    model = ""
    try:
        model = open("/proc/device-tree/model", encoding="utf-8").read().strip("\x00\n")
    except OSError:
        pass
    status = OK if arch in {"aarch64", "arm64", "AMD64", "x86_64"} else WARN
    note = "" if status == OK else " (32 bits: wheels do onnxruntime/ai-edge-litert podem não existir)"
    return status, f"{model or platform.node()} | {arch} | Python {py}{note}"


def check_config(cfg: Config, path) -> tuple[str, str]:
    detail = f"{path} | node_id={cfg.node_id}"
    if cfgmod.is_permission_too_open(path):
        return WARN, detail + " | permissão aberta demais; rode: chmod 600 " + str(path)
    return OK, detail


def check_model(cfg: Config) -> tuple[str, str]:
    m = cfg.model
    det = Detector.from_config(cfg)
    blank = np.full((cfg.camera.height, cfg.camera.width, 3), 114, np.uint8)
    det.detect(blank)  # aquecimento
    t = time.perf_counter()
    det.detect(blank)
    ms = (time.perf_counter() - t) * 1e3
    return OK, (f"{m.backend} | {m.path} | entrada {det.imgsz}x{det.imgsz} | saída {list(det.output_shape)} "
                f"({det.layout}) | {len(det.class_names)} classes | inferência {ms:.0f} ms com "
                f"{det.backend.threads} thread(s)")


def check_camera(cfg: Config) -> tuple[str, str]:
    c = cfg.camera
    cam = make_camera(c.width, c.height, c.max_fps, cfg.resolve(c.fake_images_dir) if c.fake_images_dir else None)
    cam.start()
    try:
        frame = cam.capture()
        mean = float(frame.mean())
    finally:
        cam.stop()
    kind = "FALSA (pasta)" if c.fake_images_dir else "Picamera2"
    status = WARN if mean < 20 else OK
    hint = " | imagem muito escura: tampa da lente, pouca luz ou exposição não convergiu" if status == WARN else ""
    return status, f"{kind} | quadro {frame.shape[1]}x{frame.shape[0]} | brilho médio {mean:.0f}{hint}"


def check_thermal() -> tuple[str, str]:
    sensor = VcgencmdSensor()
    if not sensor.available:
        return WARN, "vcgencmd e /sys/class/thermal indisponíveis (normal fora do Pi)"
    s = sensor.read()
    flags = s.flags.describe() if s.flags else "sem vcgencmd"
    detail = f"temp={s.temp_c} °C | throttled={flags} | arm={s.arm_freq_mhz} MHz"
    if s.flags and s.flags.under_voltage_now:
        return WARN, detail + " | SUBTENSÃO: use fonte 5,1 V/2,5 A e cabo curto"
    if s.flags and (s.flags.throttled_now or s.flags.soft_temp_limit_now):
        return WARN, detail + " | CPU estrangulada agora"
    return OK, detail


def check_network(cfg: Config) -> tuple[str, str]:
    if not cfg.server_ws_url:
        return WARN, "servidor não configurado (rode `python main.py --scan-qr`)"
    parts = urlsplit(cfg.server_ws_url)
    port = parts.port or (443 if parts.scheme == "wss" else 80)
    t = time.perf_counter()
    with socket.create_connection((parts.hostname, port), timeout=5):
        pass
    return OK, f"TCP {parts.hostname}:{port} acessível em {(time.perf_counter() - t) * 1e3:.0f} ms"


def check_server(cfg: Config) -> tuple[str, str]:
    if not cfg.server_ws_url:
        return WARN, "servidor não configurado"
    base = http_base_url(cfg.server_ws_url)
    with urllib.request.urlopen(base + "/", timeout=5) as resp:  # GET / do servidor_cv
        root = json.loads(resp.read().decode("utf-8"))
    pong = asyncio.run(_ws_ping(cfg.server_ws_url))
    return OK, f"GET / -> {root.get('message', root)!r} | WS ping -> pong {pong[:8]}…"


async def _ws_ping(url: str) -> str:
    from websockets.asyncio.client import connect

    async with connect(url, open_timeout=5) as ws:
        await ws.send('{"type":"ping"}')
        data = json.loads(await asyncio.wait_for(ws.recv(), 5))
    if data.get("type") != "pong":
        raise RuntimeError(f"resposta inesperada: {data}")
    return str(data.get("connection_id"))


def check_ws_port(cfg: Config) -> tuple[str, str]:
    w = cfg.ws_server
    if not w.enabled:
        return OK, "WebSocket do app desativado"
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((w.host, w.port))
        except OSError:
            return WARN, f"porta {w.port} em uso (o serviço pest-node já está rodando?)"
    return OK, f"porta {w.port} livre para ws://<ip-do-pi>:{w.port}{w.path}"


def check_storage(cfg: Config) -> tuple[str, str]:
    ob = Outbox(cfg.resolve(cfg.outbox.db_path), cfg.outbox.max_items, cfg.outbox.max_bytes)
    count, size = ob.stats()
    ob.close()
    free_mb = shutil.disk_usage(cfg.base_dir).free / 1024 / 1024
    mem = mem_available_mb()
    status = WARN if free_mb < 500 or (mem is not None and mem < 250) else OK
    return status, (f"fila: {count} alerta(s), {size / 1024 / 1024:.1f} MB | disco livre {free_mb:.0f} MB | "
                    f"RAM disponível {mem:.0f} MB" if mem is not None else f"disco livre {free_mb:.0f} MB")


def run_selftest(cfg: Config, path) -> int:
    checks = [
        _run("Plataforma", check_platform),
        _run("Configuração", lambda: check_config(cfg, path)),
        _run("Modelo", lambda: check_model(cfg)),
        _run("Câmera", lambda: check_camera(cfg)),
        _run("Sensores térmicos", check_thermal),
        _run("Rede", lambda: check_network(cfg)),
        _run("Servidor central", lambda: check_server(cfg)),
        _run("Porta do app", lambda: check_ws_port(cfg)),
        _run("Armazenamento/memória", lambda: check_storage(cfg)),
    ]
    print(f"\npest-node {__version__} — autoteste\n" + "=" * 72)
    for c in checks:
        print(f"[{c.status:^5}] {c.name:<22} {c.detail}")
    failed = [c for c in checks if c.status == FAIL]
    warned = [c for c in checks if c.status == WARN]
    print("=" * 72)
    print(f"Resultado: {len(checks) - len(failed) - len(warned)} OK, {len(warned)} aviso(s), {len(failed)} falha(s)\n")
    return 1 if failed else 0
