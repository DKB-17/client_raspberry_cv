"""Robustez e fidelidade do WebSocket do nó (o app Flutter conecta aqui)."""

import asyncio
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import websockets
from websockets.asyncio.client import connect

from pest_node.detector import Detector, decode_jpeg
from pest_node.models import Detection, InferenceResult
from pest_node.protocol import pack_frame
from pest_node.ws_server import AppGateway, InferenceWorker

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "teste2.onnx"
SAMPLE_ID = "48651cd1-32b5-42a3-8dbd-93aafee47985"
SAMPLE = ROOT / "tests" / "fixtures" / "sample_pest.jpg"  # Gafanhoto (dataset AgroInsect)

DET = Detection(5, "Percevejo", 0.9, (1.0, 2.0, 3.0, 4.0))


def jpeg(w=64, h=48) -> bytes:
    return cv2.imencode(".jpg", np.zeros((h, w, 3), np.uint8))[1].tobytes()


def fake_detect(delay=0.0):
    def _detect(data: bytes) -> InferenceResult:
        decode_jpeg(data)  # mesma validação do detector real
        if delay:
            time.sleep(delay)
        return InferenceResult([DET], 0.0, delay * 1000, 0.0)
    return _detect


async def gateway(detect=None, **kw):
    alerts = []

    async def on_alert(pkt, dets):
        alerts.append((pkt.packet_id, dets))

    gw = AppGateway(detect or fake_detect(), InferenceWorker(), host="127.0.0.1", port=0,
                    on_alert=on_alert, **kw)
    await gw.start()
    gw.alerts = alerts
    return gw


def url(gw):
    return f"ws://127.0.0.1:{gw.port}/ws"


async def recv_json(ws, timeout=5):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout))


async def test_ping_pong_like_app_config_page():
    gw = await gateway()
    async with connect(url(gw)) as ws:
        await ws.send('{"type":"ping"}')
        pong = await recv_json(ws)
    await gw.stop()
    assert pong["type"] == "pong" and pong["connection_id"]


async def test_photo_flow_analisando_then_processado():
    gw = await gateway()
    async with connect(url(gw)) as ws:
        await ws.send(pack_frame("foto-1", jpeg()))
        first, second = await recv_json(ws), await recv_json(ws)
    await gw.stop()
    assert first == {"id": "foto-1", "status": "analisando", "mensagem": "Imagem recebida e em análise",
                     "analysis": None, "detections": None}
    assert second["status"] == "processado"
    assert second["detections"] == [{"class_name": "Percevejo", "confidence": 0.9, "bbox": [1.0, 2.0, 3.0, 4.0]}]
    assert gw.alerts and gw.alerts[0][0] == "foto-1"


async def test_stream_has_no_analisando_and_no_alert():
    gw = await gateway()
    async with connect(url(gw)) as ws:
        await ws.send(pack_frame("STREAM_abcd1234", jpeg()))
        resp = await recv_json(ws)
    await gw.stop()
    assert resp["id"] == "STREAM_abcd1234" and resp["status"] == "processado"
    assert gw.alerts == []


async def test_stream_flood_keeps_only_latest():
    gw = await gateway(fake_detect(delay=0.3))
    async with connect(url(gw)) as ws:
        for i in range(10):
            await ws.send(pack_frame(f"STREAM_{i:08d}", jpeg()))
        got = []
        try:
            while True:
                got.append((await recv_json(ws, timeout=1.5))["id"])
        except asyncio.TimeoutError:
            pass
    await gw.stop()
    assert got[-1] == "STREAM_00000009"  # o mais recente sempre é atendido
    assert len(got) < 10  # quadros antigos foram descartados


@pytest.mark.parametrize("frame, expected_id, text", [
    (b"\x00\x00", "desconhecido", "cabeçalho"),
    (struct.pack(">I", 0) + jpeg(), "desconhecido", "zero"),
    (struct.pack(">I", 99999) + b"x", "desconhecido", "excede"),
    (struct.pack("<I", 6) + b"foto-1" + jpeg(), "desconhecido", "excede"),
    (struct.pack(">I", 50) + b"curto", "desconhecido", "truncado"),
    (pack_frame("foto-2", b"GIF89a..."), "foto-2", "JPEG"),
    (pack_frame("foto-3", b""), "foto-3", "vazia"),
])
async def test_malformed_frames_get_error_and_connection_survives(frame, expected_id, text):
    gw = await gateway()
    async with connect(url(gw)) as ws:
        await ws.send(frame)
        resp = await recv_json(ws)
        assert resp["status"] == "erro" and resp["id"] == expected_id and text in resp["mensagem"]
        await ws.send('{"type":"ping"}')  # mesma conexão continua viva
        assert (await recv_json(ws))["type"] == "pong"
    await gw.stop()


async def test_corrupted_jpeg_body_same_error_as_server():
    gw = await gateway()
    async with connect(url(gw)) as ws:
        await ws.send(pack_frame("foto-x", b"\xff\xd8" + b"\x00" * 100))
        await recv_json(ws)  # analisando
        resp = await recv_json(ws)
    await gw.stop()
    assert resp["status"] == "erro"
    assert resp["mensagem"] == "Erro no processamento pelo servidor: Não foi possível decodificar a imagem recebida."


async def test_garbage_text_rejected():
    gw = await gateway()
    async with connect(url(gw)) as ws:
        await ws.send("isto não é json")
        assert (await recv_json(ws))["status"] == "erro"
    await gw.stop()


async def test_oversized_frame_closes_only_that_connection():
    gw = await gateway(max_frame_bytes=64 * 1024)
    async with connect(url(gw), max_size=None) as ws:
        await ws.send(pack_frame("grande", b"\xff\xd8" + b"\x00" * 200_000))
        with pytest.raises(websockets.ConnectionClosed) as exc:
            await ws.recv()
        assert exc.value.rcvd.code == 1009
    async with connect(url(gw)) as ws:  # serviço continua aceitando conexões
        await ws.send('{"type":"ping"}')
        assert (await recv_json(ws))["type"] == "pong"
    await gw.stop()


async def test_disconnect_mid_message_does_not_kill_service():
    gw = await gateway()
    ws = await connect(url(gw))

    async def fragments():
        yield pack_frame("meio", b"\xff\xd8")[:10]
        ws.transport.abort()  # cai a conexão no meio de uma mensagem fragmentada
        await asyncio.sleep(10)
        yield b""

    with pytest.raises(Exception):
        await asyncio.wait_for(ws.send(fragments()), 2)
    await asyncio.sleep(0.2)
    async with connect(url(gw)) as ws2:
        await ws2.send('{"type":"ping"}')
        assert (await recv_json(ws2))["type"] == "pong"
        assert len(gw.connections) == 1  # a conexão abortada já foi liberada
    await gw.stop()


async def test_wrong_path_rejected():
    gw = await gateway()
    with pytest.raises(websockets.ConnectionClosed):
        async with connect(f"ws://127.0.0.1:{gw.port}/outro") as ws:
            await ws.recv()
    await gw.stop()


async def test_photo_queue_full_returns_error():
    gw = await gateway(fake_detect(delay=0.5), max_pending_photos=1)
    async with connect(url(gw)) as ws:
        for i in range(4):
            await ws.send(pack_frame(f"f{i}", jpeg()))
        msgs = []
        try:
            while True:
                msgs.append(await recv_json(ws, timeout=2))
        except asyncio.TimeoutError:
            pass
    await gw.stop()
    errors = [m for m in msgs if m["status"] == "erro"]
    assert errors and all("fila" in m["mensagem"] for m in errors)


@pytest.mark.skipif(not (MODEL.exists() and SAMPLE.exists()), reason="modelo ou amostra ausente")
async def test_real_model_response_matches_local_detector():
    det = Detector(MODEL)
    gw = await gateway(det.detect_jpeg)
    async with connect(url(gw)) as ws:
        await ws.send(pack_frame(SAMPLE_ID, SAMPLE.read_bytes()))
        await recv_json(ws)
        resp = await recv_json(ws, timeout=30)
    await gw.stop()
    expected = [d.to_wire() for d in det.detect_jpeg(SAMPLE.read_bytes()).detections]
    assert resp["mensagem"] == "1 detecção(ões) encontrada(s)"
    assert resp["detections"] == expected
    assert expected[0]["class_name"] == "Gafanhoto"
