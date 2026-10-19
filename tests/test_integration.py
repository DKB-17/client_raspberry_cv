"""Nó completo com câmera falsa, sensor simulado, modelo real e servidor mock."""

import asyncio
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from websockets.asyncio.client import connect

from pest_node import config as cfgmod
from pest_node.bench import main as bench_main
from pest_node.camera import ArrayCamera
from pest_node.protocol import pack_frame
from pest_node.runtime import Node
from pest_node.thermal import SimulatedSensor
from tools.mock_server import MockServer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "teste2.onnx"
SAMPLE = ROOT.parent / "servidor_cv" / "data" / "2026-09-14" / "classified" / \
    "48651cd1-32b5-42a3-8dbd-93aafee47985.jpg"

pytestmark = pytest.mark.skipif(not (MODEL.exists() and SAMPLE.exists()), reason="modelo/amostra ausentes")


def make_cfg(tmp_path, server_url=""):
    cfg = cfgmod.load(tmp_path / "config.json")
    cfg.model.path = str(MODEL)
    cfg.server_ws_url = server_url
    cfg.inference.target_fps = 5
    cfg.thermal.poll_interval_s = 1
    cfg.thermal.max_interval_s = 4
    cfg.alerts.cooldown_s = 3600
    cfg.ws_server.port = 0
    cfg.ws_server.host = "127.0.0.1"
    cfg.outbox.backoff_initial_s = 0.05
    cfg.outbox.backoff_max_s = 0.2
    return cfg


async def wait_until(cond, timeout=20.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condição não atingida")


async def test_camera_detection_raises_alert_delivered_to_server(tmp_path):
    pest = cv2.imread(str(SAMPLE))
    async with MockServer() as srv:
        node = Node(make_cfg(tmp_path, srv.url), camera=ArrayCamera([pest]),
                    sensor=SimulatedSensor([(50.0, 0)]))
        task = asyncio.create_task(node.run())
        await wait_until(lambda: node.client is not None and node.client.delivered >= 1 and node.frames >= 3)
        node.stop_event.set()
        await task
    assert srv.violations == []
    alert = [p for p in srv.received if not p.is_stream]
    assert len(alert) == 1  # cooldown impede alertas repetidos da mesma praga
    assert alert[0].image_shape == pest.shape  # imagem enviada no tamanho capturado
    assert node.frames >= 2


async def test_heat_triggers_offload_to_server_as_stream(tmp_path):
    blank = np.full((480, 640, 3), 90, np.uint8)
    async with MockServer(detections=lambda _id: []) as srv:
        node = Node(make_cfg(tmp_path, srv.url), camera=ArrayCamera([blank]),
                    sensor=SimulatedSensor([(80.0, 0)]))
        task = asyncio.create_task(node.run())
        await wait_until(lambda: node.offloaded >= 1)
        node.stop_event.set()
        await task
    assert node.controller.degraded
    assert any(p.is_stream for p in srv.received)
    assert not any(not p.is_stream for p in srv.received)  # quadro sem praga não é persistido


async def test_qr_shown_to_camera_configures_server(tmp_path):
    async with MockServer() as srv:
        qr = cv2.QRCodeEncoder.create().encode(srv.url)
        qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        qr = cv2.cvtColor(cv2.copyMakeBorder(qr, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)
        cfg = make_cfg(tmp_path)
        node = Node(cfg, camera=ArrayCamera([qr]), sensor=SimulatedSensor([(50.0, 0)]))
        task = asyncio.create_task(node.run())
        await wait_until(lambda: node.client is not None and node.client.connected)
        node.stop_event.set()
        await task
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["server_ws_url"] == srv.url


async def test_app_photo_with_pest_is_answered_and_forwarded(tmp_path):
    blank = np.full((480, 640, 3), 90, np.uint8)
    async with MockServer() as srv:
        node = Node(make_cfg(tmp_path, srv.url), camera=ArrayCamera([blank]), sensor=SimulatedSensor([(50.0, 0)]))
        task = asyncio.create_task(node.run())
        await wait_until(lambda: node.gateway is not None and node.client is not None and node.client.connected)
        async with connect(f"ws://127.0.0.1:{node.gateway.port}/ws") as ws:
            await ws.send(pack_frame("foto-app-1", SAMPLE.read_bytes()))
            first = json.loads(await ws.recv())
            second = json.loads(await asyncio.wait_for(ws.recv(), 30))
        await wait_until(lambda: node.client.delivered >= 1)
        node.stop_event.set()
        await task
    assert first["status"] == "analisando" and second["status"] == "processado"
    assert second["detections"][0]["class_name"] == "Percevejo-verde-ninfa"
    assert len([p for p in srv.received if not p.is_stream]) == 1


def test_benchmark_writes_csv_and_summary(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.ws_server.port = 8000
    cfgmod.save(cfg)
    images = tmp_path / "imgs"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), cv2.imread(str(SAMPLE)))
    out = tmp_path / "b.csv"
    assert bench_main(["--config", str(tmp_path / "config.json"), "--duration", "2", "--warmup", "1",
                       "--images", str(images), "--output", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("timestamp,elapsed_s,frame,warmup,capture_ms")
    assert len(lines) >= 3
    summary = json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert summary["frames_measured"] >= 1 and summary["fps_sustained"] > 0
