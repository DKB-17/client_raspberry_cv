"""Cliente do servidor central (WebSocket /ws), com entrega confiável de alertas.

Contrato (servidor_cv, sem nenhuma alteração):
  - Conecta em server_ws_url e faz o handshake {"type":"ping"} -> pong.
  - Alerta: envia o JPEG com id UUID v4 (NÃO "STREAM_"). O servidor responde
    "analisando" e depois "processado" (ou "erro") e PERSISTE a imagem.
    Só após "processado" o item sai da fila local (entrega "pelo menos uma vez").
  - Descarga (offload): envia o quadro com id "STREAM_xxxxxxxx"; o servidor só
    infere e devolve "processado" com as detecções, sem gravar a imagem.
  - Sem rede: reconecta com backoff exponencial; alertas aguardam na fila SQLite.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection, connect

from .logging_setup import get_logger, log_event
from .models import (STATUS_ANALYZING, STATUS_ERROR, STATUS_PROCESSED, AlertEvent, Detection)
from .outbox import Outbox
from .protocol import new_stream_id, pack_frame, ping_message

log = get_logger("client")

HANDSHAKE_TIMEOUT_S = 10.0
IDLE_POLL_S = 5.0


def backoff_delay(attempt: int, initial: float, maximum: float, jitter: float = 0.2) -> float:
    """Backoff exponencial com jitter: initial * 2^(attempt-1), limitado a maximum."""
    base = min(maximum, initial * (2 ** max(0, attempt - 1)))
    return base * (1 + random.uniform(-jitter, jitter)) if jitter else base


class ServerClient:
    def __init__(self, url: str, outbox: Outbox, class_names: list[str] | None = None,
                 max_attempts: int = 5, backoff_initial_s: float = 2.0, backoff_max_s: float = 300.0,
                 response_timeout_s: float = 60.0) -> None:
        self.url = url
        self.outbox = outbox
        self.class_names = class_names or []
        self.max_attempts = max_attempts
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s
        self.response_timeout_s = response_timeout_s
        self._ws: ClientConnection | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._wake = asyncio.Event()
        self._stopping = False
        self._task: asyncio.Task[None] | None = None
        self.connected_event = asyncio.Event()
        self.delivered = 0
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None and self.connected_event.is_set()

    # ------------------------------------------------------------------ API
    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="server-client")

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._ws is not None:
            await self._ws.close()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def enqueue_alert(self, event: AlertEvent) -> None:
        dets = json.dumps([d.to_wire() for d in event.detections], ensure_ascii=False)
        await asyncio.to_thread(self.outbox.put, event.packet_id, event.jpeg, dets, event.origin,
                                event.created_at)
        log_event(log, "alert_queued", packet_id=event.packet_id, origin=event.origin,
                  classes=[d.class_name for d in event.detections], jpeg_kb=round(len(event.jpeg) / 1024, 1))
        self._wake.set()

    async def offload(self, jpeg: bytes, timeout_s: float) -> list[Detection] | None:
        """Pede ao servidor a inferência de um quadro (sem persistir). None se falhar."""
        if not self.connected:
            return None
        packet_id = new_stream_id()
        try:
            data = await self._request(packet_id, jpeg, timeout_s)
        except (asyncio.TimeoutError, ConnectionError, websockets.ConnectionClosed) as exc:
            log_event(log, "offload_failed", level=30, error=type(exc).__name__)
            return None
        if data.get("status") != STATUS_PROCESSED:
            log_event(log, "offload_failed", level=30, error=data.get("mensagem"))
            return None
        return [Detection.from_wire(d, self.class_names) for d in data.get("detections") or []]

    # ------------------------------------------------------------ conexão
    async def _run(self) -> None:
        failures = 0
        while not self._stopping:
            try:
                async with connect(self.url, open_timeout=HANDSHAKE_TIMEOUT_S, ping_interval=20,
                                   ping_timeout=20, close_timeout=5, max_size=1024 * 1024) as ws:
                    await self._handshake(ws)
                    failures = 0
                    self._ws = ws
                    self.connected_event.set()
                    log_event(log, "server_connected", url=self.url)
                    await self._serve(ws)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # rede caída, recusa, handshake inválido...
                self.last_error = f"{type(exc).__name__}: {exc}"
                failures += 1
                if failures == 1 or failures % 10 == 0:
                    log_event(log, "server_unreachable", level=30, url=self.url, error=self.last_error,
                              failures=failures)
            finally:
                if self._ws is not None:
                    log_event(log, "server_disconnected", level=30, url=self.url)
                self._ws = None
                self.connected_event.clear()
                self._fail_pending(ConnectionError("conexão encerrada"))
            if not self._stopping:
                await asyncio.sleep(backoff_delay(max(failures, 1), self.backoff_initial_s, self.backoff_max_s))

    async def _handshake(self, ws: ClientConnection) -> None:
        await ws.send(ping_message())
        raw = await asyncio.wait_for(ws.recv(), HANDSHAKE_TIMEOUT_S)
        data = json.loads(raw)
        if data.get("type") != "pong" or not data.get("connection_id"):
            raise ConnectionError(f"handshake inválido: {str(raw)[:100]}")

    async def _serve(self, ws: ClientConnection) -> None:
        receiver = asyncio.create_task(self._receiver(ws))
        sender = asyncio.create_task(self._sender())
        done, pending = await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, websockets.ConnectionClosed):
                raise exc

    async def _receiver(self, ws: ClientConnection) -> None:
        async for message in ws:
            if isinstance(message, bytes):
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                log_event(log, "server_bad_json", level=30, sample=message[:100])
                continue
            if not isinstance(data, dict) or data.get("type") == "pong":
                continue
            status = data.get("status")
            fut = self._pending.get(str(data.get("id")))
            if status == STATUS_ANALYZING or fut is None or fut.done():
                continue
            if status in (STATUS_PROCESSED, STATUS_ERROR):
                fut.set_result(data)

    async def _request(self, packet_id: str, jpeg: bytes, timeout_s: float) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise ConnectionError("sem conexão com o servidor")
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[packet_id] = fut
        try:
            await ws.send(pack_frame(packet_id, jpeg))
            return await asyncio.wait_for(fut, timeout_s)
        finally:
            self._pending.pop(packet_id, None)

    def _fail_pending(self, exc: Exception) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    # ------------------------------------------------------------ entrega
    async def _sender(self) -> None:
        while not self._stopping:
            item = await asyncio.to_thread(self.outbox.next_due)
            if item is None:
                wait = await asyncio.to_thread(self.outbox.seconds_until_next)
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), min(wait if wait is not None else IDLE_POLL_S,
                                                                  IDLE_POLL_S))
                except asyncio.TimeoutError:
                    pass
                continue
            await self._deliver(item.packet_id, item.jpeg, item.attempts, item.created_at)

    async def _deliver(self, packet_id: str, jpeg: bytes, attempts: int, created_at: float) -> None:
        try:
            data = await self._request(packet_id, jpeg, self.response_timeout_s)
        except asyncio.TimeoutError:
            await self._retry(packet_id, attempts, "sem resposta do servidor (timeout)")
            return
        if data.get("status") == STATUS_PROCESSED:
            await asyncio.to_thread(self.outbox.delete, packet_id)
            self.delivered += 1
            log_event(log, "alert_delivered", packet_id=packet_id,
                      server_detections=len(data.get("detections") or []),
                      latency_s=round(time.time() - created_at, 1), attempts=attempts + 1)
        else:
            await self._retry(packet_id, attempts, str(data.get("mensagem")))

    async def _retry(self, packet_id: str, attempts: int, error: str) -> None:
        if attempts + 1 >= self.max_attempts:
            await asyncio.to_thread(self.outbox.delete, packet_id)
            log_event(log, "alert_discarded", level=40, packet_id=packet_id, attempts=attempts + 1, error=error)
            return
        delay = backoff_delay(attempts + 1, self.backoff_initial_s, self.backoff_max_s)
        await asyncio.to_thread(self.outbox.mark_failed, packet_id, time.time() + delay, error)
        log_event(log, "alert_retry", level=30, packet_id=packet_id, attempt=attempts + 1,
                  retry_in_s=round(delay, 1), error=error)
