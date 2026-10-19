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
from pest_node.camera import ArrayCamera
from pest_node.protocol import pack_frame
from main import executar
from pest_node.thermal import SimulatedSensor
from tools.mock_server import MockServer

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "teste2.onnx"
SAMPLE = ROOT / "tests" / "fixtures" / "sample_pest.jpg"  # Gafanhoto (dataset AgroInsect)

pytestmark = pytest.mark.skipif(not (MODEL.exists() and SAMPLE.exists()), reason="modelo/amostra ausentes")


def make_cfg(tmp_path, server_url=""):
    cfg = cfgmod.load(tmp_path / "config.json")
    cfg.model.path = str(MODEL)
    cfg.server_ws_url = server_url
    cfg.inference.capture_interval_s = 0.5
    cfg.alerts.cooldown_s = 3600
    cfg.ws_server.port = 0
    cfg.ws_server.host = "127.0.0.1"
    cfg.outbox.backoff_initial_s = 0.05
    cfg.outbox.backoff_max_s = 0.2
    return cfg


def start(cfg, camera, sensor, forcar_qr=False):
    """Roda o fluxo completo do main.py em segundo plano; devolve (tarefa, nó)."""
    holder = {}
    task = asyncio.create_task(executar(cfg, forcar_qr=forcar_qr, camera=camera, sensor=sensor,
                                        on_ready=lambda n: holder.setdefault("node", n)))
    return task, holder


async def started(holder):
    await wait_until(lambda: "node" in holder)
    return holder["node"]


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
        task, holder = start(make_cfg(tmp_path, srv.url), ArrayCamera([pest]), SimulatedSensor([(50.0, 0)]))
        node = await started(holder)
        await wait_until(lambda: node.client is not None and node.client.delivered >= 1 and node.frames >= 3)
        node.stop_event.set()
        await task
    assert srv.violations == []
    alert = [p for p in srv.received if not p.is_stream]
    assert len(alert) == 1  # cooldown impede alertas repetidos da mesma praga
    assert alert[0].image_shape == pest.shape  # imagem enviada no tamanho capturado
    assert node.frames >= 2


def qr_frame(text):
    qr = cv2.QRCodeEncoder.create().encode(text)
    qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(cv2.copyMakeBorder(qr, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)


async def test_qr_shown_to_camera_configures_server(tmp_path):
    async with MockServer() as srv:
        qr = cv2.QRCodeEncoder.create().encode(srv.url)
        qr = cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        qr = cv2.cvtColor(cv2.copyMakeBorder(qr, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)
        cfg = make_cfg(tmp_path)
        task, holder = start(cfg, ArrayCamera([qr]), SimulatedSensor([(50.0, 0)]))
        node = await started(holder)
        await wait_until(lambda: node.client is not None and node.client.connected)
        node.stop_event.set()
        await task
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["server_ws_url"] == srv.url


async def test_qr_capture_interval_drives_collection(tmp_path):
    """QR no formato novo do servidor: a URL e o intervalo de captura vão para o config.json."""
    async with MockServer() as srv:
        cfg = make_cfg(tmp_path)
        cfg.inference.capture_interval_s = 60
        frames = [qr_frame(f"{srv.url}?capture_interval=0.5")] + [np.full((480, 640, 3), 90, np.uint8)]
        task, holder = start(cfg, ArrayCamera(frames), SimulatedSensor([(50.0, 0)]))
        node = await started(holder)
        await wait_until(lambda: node.frames >= 3, timeout=10)  # com 60 s não daria 3 quadros
        node.stop_event.set()
        await task
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert saved["server_ws_url"] == srv.url
    assert saved["inference"]["capture_interval_s"] == 0.5
    assert node.interval_s == 0.5


async def test_mode_server_sends_every_frame_for_analysis(tmp_path):
    blank = np.full((480, 640, 3), 90, np.uint8)
    async with MockServer(detections=lambda _id: []) as srv:
        cfg = make_cfg(tmp_path, srv.url)
        cfg.inference.mode = "server"
        task, holder = start(cfg, ArrayCamera([blank]), SimulatedSensor([(50.0, 0)]))
        node = await started(holder)
        await wait_until(lambda: node.offloaded >= 3)
        node.stop_event.set()
        await task
    assert all(p.is_stream for p in srv.received)


async def test_mode_local_never_sends_frames_even_when_hot(tmp_path):
    """Sem ajuste por temperatura: mesmo a 80 °C o intervalo e o modo não mudam."""
    blank = np.full((480, 640, 3), 90, np.uint8)
    async with MockServer() as srv:
        cfg = make_cfg(tmp_path, srv.url)
        cfg.inference.mode = "local"
        task, holder = start(cfg, ArrayCamera([blank]), SimulatedSensor([(80.0, 0)]))
        node = await started(holder)
        await wait_until(lambda: node.frames >= 2)
        node.stop_event.set()
        await task
    assert node.offloaded == 0 and srv.received == []
    assert node.interval_s == cfg.inference.capture_interval_s


async def test_app_photo_with_pest_is_answered_and_forwarded(tmp_path):
    blank = np.full((480, 640, 3), 90, np.uint8)
    async with MockServer() as srv:
        task, holder = start(make_cfg(tmp_path, srv.url), ArrayCamera([blank]), SimulatedSensor([(50.0, 0)]))
        node = await started(holder)
        await wait_until(lambda: node.gateway is not None and node.client is not None and node.client.connected)
        async with connect(f"ws://127.0.0.1:{node.gateway.port}/ws") as ws:
            await ws.send(pack_frame("foto-app-1", SAMPLE.read_bytes()))
            first = json.loads(await ws.recv())
            second = json.loads(await asyncio.wait_for(ws.recv(), 30))
        await wait_until(lambda: node.client.delivered >= 1)
        node.stop_event.set()
        await task
    assert first["status"] == "analisando" and second["status"] == "processado"
    assert second["detections"][0]["class_name"] == "Gafanhoto"
    assert len([p for p in srv.received if not p.is_stream]) == 1

