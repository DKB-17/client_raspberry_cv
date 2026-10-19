import asyncio
import json
import socket
import time

import cv2
import numpy as np
import pytest

from pest_node.client import ServerClient, backoff_delay
from pest_node.models import AlertEvent, Detection
from pest_node.outbox import Outbox
from pest_node.protocol import new_packet_id
from tools.mock_server import MockServer

DET = Detection(5, "Percevejo", 0.88, (10.0, 20.0, 30.0, 40.0))


def jpeg(seed=0) -> bytes:
    img = np.full((240, 320, 3), seed % 255, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def wait_until(cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("condição não atingida a tempo")


def make_client(tmp_path, url, **kw):
    ob = Outbox(tmp_path / "outbox.sqlite", max_items=kw.pop("max_items", 50))
    params = dict(backoff_initial_s=0.05, backoff_max_s=0.2, response_timeout_s=2.0)
    params.update(kw)
    return ServerClient(url, ob, class_names=["Percevejo"], **params), ob


# ------------------------------------------------------------- backoff
def test_backoff_exponential_and_capped():
    vals = [backoff_delay(n, 2, 300, jitter=0) for n in range(1, 11)]
    assert vals[:5] == [2, 4, 8, 16, 32]
    assert max(vals) == 300
    assert 1.6 <= backoff_delay(1, 2, 300) <= 2.4


# ------------------------------------------------------------- outbox
def test_outbox_limits_drop_oldest(tmp_path):
    ob = Outbox(tmp_path / "o.sqlite", max_items=3)
    for i in range(5):
        ob.put(f"id{i}", jpeg(i), "[]", "camera", now=100 + i)
    assert ob.stats()[0] == 3
    assert ob.next_due(now=1000).packet_id == "id2"


def test_outbox_persists_across_reopen(tmp_path):
    ob = Outbox(tmp_path / "o.sqlite")
    ob.put("abc", jpeg(), "[]", "camera")
    ob.close()
    assert Outbox(tmp_path / "o.sqlite").next_due().packet_id == "abc"


# ------------------------------------------------------------- contrato com o mock
async def test_alert_delivered_with_contract(tmp_path):
    async with MockServer() as srv:
        client, ob = make_client(tmp_path, srv.url)
        await client.start()
        pid = new_packet_id()
        await client.enqueue_alert(AlertEvent(pid, time.time(), jpeg(), [DET]))
        await wait_until(lambda: client.delivered == 1)
        await client.stop()
    assert srv.pings >= 1
    assert srv.violations == []
    assert [(p.packet_id, p.is_stream) for p in srv.received] == [(pid, False)]
    assert srv.received[0].image_shape == (240, 320, 3)
    assert ob.stats() == (0, 0)


async def test_offline_queue_then_delivery_when_server_returns(tmp_path):
    port = free_port()
    client, ob = make_client(tmp_path, f"ws://127.0.0.1:{port}/ws")
    await client.start()
    ids = [new_packet_id() for _ in range(3)]
    for pid in ids:
        await client.enqueue_alert(AlertEvent(pid, time.time(), jpeg(), [DET]))
    await asyncio.sleep(0.5)
    assert not client.connected and ob.stats()[0] == 3  # nada perdido offline
    async with MockServer(port=port) as srv:
        await wait_until(lambda: client.delivered == 3)
        await client.stop()
    assert sorted(p.packet_id for p in srv.received) == sorted(ids)
    assert ob.stats()[0] == 0


async def test_server_error_retries_then_discards(tmp_path):
    async with MockServer() as srv:
        pid = new_packet_id()
        srv.error_ids.add(pid)
        client, ob = make_client(tmp_path, srv.url, max_attempts=3)
        await client.start()
        await client.enqueue_alert(AlertEvent(pid, time.time(), jpeg(), [DET]))
        await wait_until(lambda: ob.stats()[0] == 0)
        await client.stop()
    assert [p.packet_id for p in srv.received] == [pid] * 3


async def test_connection_drop_mid_request_keeps_item(tmp_path):
    async with MockServer() as srv:
        pid = new_packet_id()
        srv.drop_ids.add(pid)  # primeira tentativa: servidor fecha sem responder
        client, ob = make_client(tmp_path, srv.url)
        await client.start()
        await client.enqueue_alert(AlertEvent(pid, time.time(), jpeg(), [DET]))
        await wait_until(lambda: client.delivered == 1)
        await client.stop()
    assert [p.packet_id for p in srv.received] == [pid, pid]


async def test_offload_uses_stream_id_and_returns_detections(tmp_path):
    async with MockServer() as srv:
        client, _ = make_client(tmp_path, srv.url)
        await client.start()
        await wait_until(lambda: client.connected)
        dets = await client.offload(jpeg(), timeout_s=2)
        await client.stop()
    assert dets and dets[0].class_name == "Percevejo-verde-ninfa"
    assert srv.received[0].is_stream and srv.received[0].packet_id.startswith("STREAM_")


async def test_offload_returns_none_when_offline(tmp_path):
    client, _ = make_client(tmp_path, f"ws://127.0.0.1:{free_port()}/ws")
    assert await client.offload(jpeg(), timeout_s=1) is None


async def test_queued_detections_json_is_wire_format(tmp_path):
    client, ob = make_client(tmp_path, "ws://127.0.0.1:1/ws")
    await client.enqueue_alert(AlertEvent("x", time.time(), jpeg(), [DET]))
    item = ob.next_due()
    assert json.loads(item.detections) == [{"class_name": "Percevejo", "confidence": 0.88,
                                            "bbox": [10.0, 20.0, 30.0, 40.0]}]
