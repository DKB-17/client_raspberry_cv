"""Orquestração do nó: câmera -> inferência -> alertas, app WebSocket, térmico, métricas."""

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
from .discovery import InvalidServerUrl, decode_qr, normalize_server_url
from .logging_setup import get_logger, log_event
from .models import AlertEvent, Detection
from .outbox import Outbox
from .protocol import ImagePacket, new_packet_id
from .sysmetrics import CpuMeter, mem_available_mb, process_rss_mb
from .thermal import AdaptiveController, ThermalSensor, VcgencmdSensor
from .ws_server import AppGateway, InferenceWorker

log = get_logger("runtime")

CAMERA_RETRY_S = 30.0


class Node:
    def __init__(self, cfg: Config, camera: Camera | None = None, sensor: ThermalSensor | None = None,
                 detector: Detector | None = None) -> None:
        self.cfg = cfg
        self.camera = camera
        self.sensor = sensor or VcgencmdSensor()
        self.detector = detector
        self.controller = AdaptiveController(
            target_fps=cfg.inference.target_fps, temp_high_c=cfg.thermal.temp_high_c,
            temp_resume_c=cfg.thermal.temp_resume_c, slowdown_factor=cfg.thermal.slowdown_factor,
            max_interval_s=cfg.thermal.max_interval_s, react_to_throttling=cfg.thermal.react_to_throttling,
            react_to_under_voltage=cfg.thermal.react_to_under_voltage)
        self.policy = AlertPolicy(cfg.alerts.threshold, cfg.alerts.cooldown_s, cfg.alerts.classes)
        self.worker = InferenceWorker()
        self.outbox: Outbox | None = None
        self.client: ServerClient | None = None
        self.gateway: AppGateway | None = None
        self.stop_event = asyncio.Event()
        # métricas (janela limitada; nada cresce sem limite)
        self.frames = 0
        self.offloaded = 0
        self.alerts = 0
        self._infer_ms: deque[float] = deque(maxlen=200)
        self._e2e_ms: deque[float] = deque(maxlen=200)
        self._cpu = CpuMeter()

    # ------------------------------------------------------------ ciclo de vida
    async def run(self) -> None:
        cfg = self.cfg
        if self.detector is None:
            m = cfg.model
            self.detector = await asyncio.to_thread(
                Detector, cfg.resolve(m.path), m.conf_threshold, m.iou_threshold, m.max_detections,
                m.intra_op_num_threads, m.inter_op_num_threads)
        self.outbox = Outbox(cfg.resolve(cfg.outbox.db_path), cfg.outbox.max_items, cfg.outbox.max_bytes)
        if cfg.server_ws_url:
            await self._start_client(cfg.server_ws_url)
        else:
            log_event(log, "server_not_configured", level=30,
                      hint="mostre o QR do servidor à câmera ou rode `pest-node scan-qr`; alertas ficam na fila")
        if cfg.ws_server.enabled:
            w = cfg.ws_server
            self.gateway = AppGateway(self.detector.detect_jpeg, self.worker, w.host, w.port, w.path,
                                      w.max_frame_bytes, w.max_id_bytes, w.max_pending_photos,
                                      on_alert=self._on_app_photo if w.forward_alerts else None)
            await self.gateway.start()
        tasks = [asyncio.create_task(self._camera_loop(), name="camera"),
                 asyncio.create_task(self._thermal_loop(), name="thermal"),
                 asyncio.create_task(self._metrics_loop(), name="metrics")]
        log_event(log, "node_started", node_id=cfg.node_id, target_fps=cfg.inference.target_fps,
                  threads=cfg.model.intra_op_num_threads, server=cfg.server_ws_url or None)
        try:
            await self.stop_event.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._shutdown()

    async def _shutdown(self) -> None:
        if self.gateway:
            await self.gateway.stop()
        if self.client:
            await self.client.stop()
        if self.camera:
            await asyncio.to_thread(self.camera.stop)
        self.worker.shutdown()
        if self.outbox:
            self.outbox.close()
        log_event(log, "node_stopped", frames=self.frames, alerts=self.alerts)

    async def _start_client(self, url: str) -> None:
        assert self.outbox is not None and self.detector is not None
        o = self.cfg.outbox
        self.client = ServerClient(url, self.outbox, self.detector.class_names, o.max_attempts,
                                   o.backoff_initial_s, o.backoff_max_s, o.response_timeout_s)
        await self.client.start()

    # ------------------------------------------------------------ câmera
    async def _open_camera(self) -> Camera | None:
        c = self.cfg.camera
        if self.camera is None:
            fake = self.cfg.resolve(c.fake_images_dir) if c.fake_images_dir else None
            self.camera = make_camera(c.width, c.height, c.max_fps, fake)
        try:
            await asyncio.to_thread(self.camera.start)
            return self.camera
        except CameraError as exc:
            log_event(log, "camera_unavailable", level=40, error=str(exc), retry_in_s=CAMERA_RETRY_S)
            self.camera = None
            return None

    async def _camera_loop(self) -> None:
        camera = None
        while camera is None:
            camera = await self._open_camera()
            if camera is None:
                await asyncio.sleep(CAMERA_RETRY_S)  # o WebSocket do app segue funcionando
        while True:
            started = time.monotonic()
            try:
                await self._process_frame(camera)
            except CameraError as exc:
                log_event(log, "camera_error", level=40, error=str(exc))
                await asyncio.sleep(5)
            except Exception:
                log.exception("erro inesperado no ciclo da câmera")
                await asyncio.sleep(5)
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, self.controller.interval_s - elapsed))

    async def _process_frame(self, camera: Camera) -> None:
        t0 = time.perf_counter()
        frame = await asyncio.to_thread(camera.capture)
        if not self.cfg.server_ws_url:
            await self._try_discover(frame)
        jpeg: bytes | None = None
        detections: list[Detection] | None = None
        source = "local"
        if self.controller.degraded and self.cfg.inference.offload_when_degraded and self.client \
                and self.client.connected:
            jpeg = await asyncio.to_thread(encode_jpeg, frame, self.cfg.camera.jpeg_quality)
            detections = await self.client.offload(jpeg, self.cfg.inference.offload_timeout_s)
            if detections is not None:
                source = "server"
                self.offloaded += 1
        if detections is None:
            assert self.detector is not None
            result = await self.worker.run_camera(self.detector.detect, frame)
            detections = result.detections
            self._infer_ms.append(result.inference_ms)
        self.frames += 1
        self._e2e_ms.append((time.perf_counter() - t0) * 1e3)
        if detections:
            log_event(log, "detections", source=source,
                      found=[f"{d.class_name}:{d.confidence:.2f}" for d in detections])
        hits = self.policy.triggering(detections)
        if hits:
            if jpeg is None:
                jpeg = await asyncio.to_thread(encode_jpeg, frame, self.cfg.camera.jpeg_quality)
            await self._raise_alert(jpeg, detections, "camera")

    async def _try_discover(self, frame: np.ndarray) -> None:
        text = await asyncio.to_thread(decode_qr, frame)
        if not text:
            return
        try:
            url = normalize_server_url(text)
        except InvalidServerUrl as exc:
            log_event(log, "qr_invalid", level=30, reason=str(exc))
            return
        self.cfg.server_ws_url = url
        await asyncio.to_thread(cfgmod.save, self.cfg)
        log_event(log, "server_discovered", url=url)
        await self._start_client(url)

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
        else:
            assert self.outbox is not None
            dets = json.dumps([d.to_wire() for d in detections], ensure_ascii=False)
            await asyncio.to_thread(self.outbox.put, event.packet_id, jpeg, dets, origin, event.created_at)

    # ------------------------------------------------------------ térmico e métricas
    async def _thermal_loop(self) -> None:
        while True:
            sample = await asyncio.to_thread(self.sensor.read)
            self.controller.update(sample)
            await asyncio.sleep(self.cfg.thermal.poll_interval_s)

    async def _metrics_loop(self) -> None:
        last_frames, last_t = self.frames, time.monotonic()
        while True:
            await asyncio.sleep(self.cfg.logging.metrics_interval_s)
            now = time.monotonic()
            fps = (self.frames - last_frames) / (now - last_t)
            last_frames, last_t = self.frames, now
            log_event(log, "metrics", **self.snapshot(fps))

    def snapshot(self, fps: float | None = None) -> dict[str, object]:
        s = self.controller.last_sample
        infer = list(self._infer_ms)
        e2e = list(self._e2e_ms)
        count, size = self.outbox.stats() if self.outbox else (0, 0)
        return {
            "fps": None if fps is None else round(fps, 3),
            "infer_ms_avg": round(statistics.fmean(infer), 1) if infer else None,
            "infer_ms_p95": round(_p95(infer), 1) if infer else None,
            "e2e_ms_avg": round(statistics.fmean(e2e), 1) if e2e else None,
            "interval_s": round(self.controller.interval_s, 2),
            "load_reason": self.controller.reason,
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
