"""Serviços do nó usados pelo main.py.

O main.py conduz o fluxo (câmera -> QR -> coleta). Esta classe reúne as peças:
modelo, fila de alertas, cliente do servidor, WebSocket do app e métricas, e expõe `collect_once()`, que é UM ciclo da coleta:
captura -> análise (YOLO no Pi ou no servidor) -> alerta.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections import deque

import numpy as np

from . import config as cfgmod
from .alerts import AlertPolicy
from .camera import Camera, CameraError, make_camera
from .client import ServerClient
from .config import Config
from .detector import Detector, encode_jpeg
from .discovery import InvalidServerUrl, ServerInfo, decode_qr, parse_server_qr
from .logging_setup import get_logger, log_event
from .models import AlertEvent, Detection
from .outbox import Outbox
from .protocol import ImagePacket, new_packet_id
from .sysmetrics import CpuMeter, mem_available_mb, process_rss_mb
from .thermal import ThermalSample, ThermalSensor, VcgencmdSensor
from .ws_server import AppGateway, InferenceWorker

log = get_logger("runtime")


class Node:
    def __init__(self, cfg: Config, camera: Camera | None = None, sensor: ThermalSensor | None = None,
                 detector: Detector | None = None) -> None:
        self.cfg = cfg
        c = cfg.camera
        self.camera = camera or make_camera(c.width, c.height, c.max_fps,
                                            cfg.resolve(c.fake_images_dir) if c.fake_images_dir else None)
        self.sensor = sensor or VcgencmdSensor()
        self.detector = detector
        self.policy = AlertPolicy(cfg.alerts.threshold, cfg.alerts.cooldown_s, cfg.alerts.classes)
        self.worker = InferenceWorker()
        self.outbox: Outbox | None = None
        self.client: ServerClient | None = None
        self.gateway: AppGateway | None = None
        self.stop_event = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self._camera_open = False
        # métricas (janelas limitadas; nada cresce sem limite)
        self.frames = 0
        self.offloaded = 0
        self.alerts = 0
        self._infer_ms: deque[float] = deque(maxlen=200)
        self._e2e_ms: deque[float] = deque(maxlen=200)
        self._cpu = CpuMeter()

    @property
    def interval_s(self) -> float:
        """Segundos entre capturas (inference.capture_interval_s, vindo do QR do servidor)."""
        return self.cfg.inference.capture_interval_s

    # ------------------------------------------------------------ ciclo de vida
    async def start(self) -> None:
        """Carrega o modelo e sobe fila, cliente, WebSocket do app e métricas."""
        cfg = self.cfg
        if self.detector is None:
            self.detector = await asyncio.to_thread(Detector.from_config, cfg)
        self.outbox = Outbox(cfg.resolve(cfg.outbox.db_path), cfg.outbox.max_items, cfg.outbox.max_bytes)
        if cfg.server_ws_url:
            await self._start_client(cfg.server_ws_url)
        if cfg.ws_server.enabled:
            w = cfg.ws_server
            self.gateway = AppGateway(self.detector.detect_jpeg, self.worker, w.host, w.port, w.path,
                                      w.max_frame_bytes, w.max_id_bytes, w.max_pending_photos,
                                      on_alert=self._on_app_photo if w.forward_alerts else None)
            await self.gateway.start()
        self._tasks = [asyncio.create_task(self._metrics_loop(), name="metrics")]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.gateway:
            await self.gateway.stop()
        if self.client:
            await self.client.stop()
        if self._camera_open:
            await asyncio.to_thread(self.camera.stop)
            self._camera_open = False
        self.worker.shutdown()
        if self.outbox:
            self.outbox.close()
        log_event(log, "node_stopped", frames=self.frames, alerts=self.alerts)

    async def open_camera(self) -> bool:
        try:
            await asyncio.to_thread(self.camera.start)
        except CameraError as exc:
            log_event(log, "camera_unavailable", level=40, error=str(exc))
            return False
        self._camera_open = True
        return True

    async def _start_client(self, url: str) -> None:
        assert self.outbox is not None and self.detector is not None
        o = self.cfg.outbox
        self.client = ServerClient(url, self.outbox, self.detector.class_names, o.max_attempts,
                                   o.backoff_initial_s, o.backoff_max_s, o.response_timeout_s)
        await self.client.start()

    async def use_server(self, info: ServerInfo) -> None:
        """Aplica o que veio do QR: URL do servidor e intervalo de captura. Grava no config.json."""
        self.cfg.server_ws_url = info.url
        if info.capture_interval_s is not None:
            self.cfg.inference.capture_interval_s = info.capture_interval_s
        await asyncio.to_thread(cfgmod.save, self.cfg)
        log_event(log, "server_configured", url=info.url, capture_interval_s=self.cfg.inference.capture_interval_s)
        if self.client is not None:
            await self.client.stop()
        await self._start_client(info.url)

    # ------------------------------------------------------------ um ciclo da coleta
    async def collect_once(self) -> list[Detection]:
        """Captura um quadro, analisa com o YOLO e dispara alerta se houver praga."""
        t0 = time.perf_counter()
        frame = await asyncio.to_thread(self.camera.capture)
        if not self.cfg.server_ws_url:
            await self._try_discover(frame)
        detections, source, jpeg = await self._analyze(frame)
        self.frames += 1
        self._e2e_ms.append((time.perf_counter() - t0) * 1e3)
        if detections:
            log_event(log, "detections", source=source,
                      found=[f"{d.class_name}:{d.confidence:.2f}" for d in detections])
        if self.policy.triggering(detections):
            if jpeg is None:
                jpeg = await asyncio.to_thread(encode_jpeg, frame, self.cfg.camera.jpeg_quality)
            await self._raise_alert(jpeg, detections, "camera")
        return detections

    def _send_to_server(self) -> bool:
        connected = self.client is not None and self.client.connected
        return self.cfg.inference.mode == "server" and connected

    async def _analyze(self, frame: np.ndarray) -> tuple[list[Detection], str, bytes | None]:
        """YOLO no servidor (quadro STREAM_, não é gravado) ou no Pi, conforme inference.mode."""
        jpeg: bytes | None = None
        if self._send_to_server():
            assert self.client is not None
            jpeg = await asyncio.to_thread(encode_jpeg, frame, self.cfg.camera.jpeg_quality)
            detections = await self.client.offload(jpeg, self.cfg.inference.server_timeout_s)
            if detections is not None:
                self.offloaded += 1
                return detections, "server", jpeg
        assert self.detector is not None
        result = await self.worker.run_camera(self.detector.detect, frame)
        self._infer_ms.append(result.inference_ms)
        return result.detections, "local", jpeg

    async def _try_discover(self, frame: np.ndarray) -> None:
        text = await asyncio.to_thread(decode_qr, frame)
        if not text:
            return
        try:
            info = parse_server_qr(text)
        except InvalidServerUrl as exc:
            log_event(log, "qr_invalid", level=30, reason=str(exc))
            return
        await self.use_server(info)

    # ------------------------------------------------------------ alertas
    async def _on_app_photo(self, packet: ImagePacket, detections: list[Detection]) -> None:
        if self.policy.triggering(detections):
            await self._raise_alert(packet.jpeg, detections, "app")

    async def _raise_alert(self, jpeg: bytes, detections: list[Detection], origin: str) -> None:
        event = AlertEvent(new_packet_id(), time.time(), jpeg, detections, origin)  # type: ignore[arg-type]
        self.alerts += 1
        log_event(log, "alert", level=30, packet_id=event.packet_id, origin=origin,
                  classes=sorted({d.class_name for d in detections if d.confidence >= self.policy.threshold}))
        if self.client is not None:
            await self.client.enqueue_alert(event)
        else:  # servidor ainda não configurado: fica na fila até o QR ser lido
            assert self.outbox is not None
            dets = json.dumps([d.to_wire() for d in detections], ensure_ascii=False)
            await asyncio.to_thread(self.outbox.put, event.packet_id, jpeg, dets, origin, event.created_at)

    # ------------------------------------------------------------ métricas
    async def _metrics_loop(self) -> None:
        last_frames, last_t = self.frames, time.monotonic()
        while True:
            await asyncio.sleep(self.cfg.logging.metrics_interval_s)
            now = time.monotonic()
            fps = (self.frames - last_frames) / (now - last_t)
            last_frames, last_t = self.frames, now
            sample = await asyncio.to_thread(self.sensor.read)  # só registra; não altera a coleta
            log_event(log, "metrics", **self.snapshot(fps, sample))

    def snapshot(self, fps: float | None = None, s: ThermalSample | None = None) -> dict[str, object]:
        infer = list(self._infer_ms)
        e2e = list(self._e2e_ms)
        count, size = self.outbox.stats() if self.outbox else (0, 0)
        return {
            "fps": None if fps is None else round(fps, 3),
            "infer_ms_avg": round(statistics.fmean(infer), 1) if infer else None,
            "infer_ms_p95": round(_p95(infer), 1) if infer else None,
            "e2e_ms_avg": round(statistics.fmean(e2e), 1) if e2e else None,
            "capture_interval_s": self.cfg.inference.capture_interval_s,
            "mode": self.cfg.inference.mode,
            "temp_c": s.temp_c if s else None,
            "throttled": s.flags.describe() if s and s.flags else None,
            "arm_mhz": s.arm_freq_mhz if s else None,
            "cpu_pct": _round(self._cpu.percent()),
            "rss_mb": _round(process_rss_mb()),
            "mem_available_mb": _round(mem_available_mb()),
            "frames": self.frames,
            "offloaded": self.offloaded,
            "alerts": self.alerts,
            "outbox_items": count,
            "outbox_mb": round(size / 1024 / 1024, 2),
            "server_connected": bool(self.client and self.client.connected),
            "app_connections": len(self.gateway.connections) if self.gateway else 0,
        }


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 1)
