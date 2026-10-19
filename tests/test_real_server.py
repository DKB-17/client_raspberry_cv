"""Contrato contra o servidor_cv REAL (código sem alterações).

Suba o servidor e informe a URL, por exemplo:
    cd servidor_cv && uvicorn src.main:app --host 127.0.0.1 --port 8765
    REAL_SERVER_URL=ws://127.0.0.1:8765/ws pytest tests/test_real_server.py
Sem REAL_SERVER_URL os testes são pulados. Use DATA_DIR no servidor apontando para
uma pasta temporária para não misturar imagens de teste com os dados reais.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from websockets.asyncio.client import connect

from pest_node.client import ServerClient
from pest_node.detector import Detector
from pest_node.models import AlertEvent
from pest_node.outbox import Outbox
from pest_node.protocol import new_packet_id, pack_frame

URL = os.environ.get("REAL_SERVER_URL", "")
ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ID = "48651cd1-32b5-42a3-8dbd-93aafee47985"
SAMPLE = ROOT.parent / "servidor_cv" / "data" / "2026-09-14" / "classified" / f"{SAMPLE_ID}.jpg"

pytestmark = [pytest.mark.real_server, pytest.mark.skipif(not URL, reason="REAL_SERVER_URL não definido")]


async def test_handshake_ping_pong():
    async with connect(URL) as ws:
        await ws.send('{"type":"ping"}')
        data = json.loads(await asyncio.wait_for(ws.recv(), 10))
    assert data["type"] == "pong" and data["connection_id"]


async def test_alert_is_accepted_and_persisted_by_real_server(tmp_path):
    client = ServerClient(URL, Outbox(tmp_path / "o.sqlite"), response_timeout_s=60,
                          backoff_initial_s=0.2, backoff_max_s=1)
    await client.start()
    pid = new_packet_id()
    await client.enqueue_alert(AlertEvent(pid, time.time(), SAMPLE.read_bytes(), []))
    end = time.monotonic() + 60
    while client.delivered < 1 and time.monotonic() < end:
        await asyncio.sleep(0.1)
    await client.stop()
    assert client.delivered == 1
    data_dir = os.environ.get("REAL_SERVER_DATA_DIR")
    if data_dir:  # confere que o servidor gravou a imagem em classified/
        saved = list(Path(data_dir).glob(f"*/classified/{pid}.jpg"))
        assert saved and saved[0].read_bytes() == SAMPLE.read_bytes()


async def test_offload_stream_frame_and_local_equals_server():
    async with connect(URL) as ws:
        await ws.send(pack_frame("STREAM_0000abcd", SAMPLE.read_bytes()))
        resp = json.loads(await asyncio.wait_for(ws.recv(), 60))  # sem "analisando" para STREAM_
    assert resp["status"] == "processado" and resp["id"] == "STREAM_0000abcd"
    local = [d.to_wire() for d in Detector(ROOT / "models" / "teste2.onnx").detect_jpeg(SAMPLE.read_bytes()).detections]
    assert [d["class_name"] for d in local] == [d["class_name"] for d in resp["detections"]]
    for a, b in zip(local, resp["detections"]):
        assert a["confidence"] == pytest.approx(b["confidence"], abs=0.01)
        np.testing.assert_allclose(a["bbox"], b["bbox"], atol=2)


async def test_node_and_server_reply_identically_to_same_frame():
    """O app recebe do nó o mesmo JSON (exceto ids) que receberia do servidor."""
    from pest_node.ws_server import AppGateway, InferenceWorker

    det = Detector(ROOT / "models" / "teste2.onnx")
    gw = AppGateway(det.detect_jpeg, InferenceWorker(), host="127.0.0.1", port=0)
    await gw.start()
    frame = pack_frame(SAMPLE_ID, SAMPLE.read_bytes())
    replies = []
    for url in (URL, f"ws://127.0.0.1:{gw.port}/ws"):
        async with connect(url) as ws:
            await ws.send(frame)
            replies.append([json.loads(await asyncio.wait_for(ws.recv(), 60)) for _ in range(2)])
    await gw.stop()
    server, node = replies
    assert [m["status"] for m in server] == [m["status"] for m in node] == ["analisando", "processado"]
    assert server[0] == node[0]
    assert set(server[1]) == set(node[1])
    assert server[1]["mensagem"] == node[1]["mensagem"]
    for a, b in zip(server[1]["detections"], node[1]["detections"]):
        assert a["class_name"] == b["class_name"]
        assert a["confidence"] == pytest.approx(b["confidence"], abs=0.01)
