"""WebSocket para o app Flutter: clone do /ws do servidor central, servido pelo Pi.

O app conecta em ws://<ip-do-pi>:8000/ws sem nenhuma alteração: mesmo
handshake (ping/pong), mesmo quadro binário e mesmas respostas JSON.

Robustez:
  - quadros acima de max_frame_bytes: a biblioteca fecha só aquela conexão (1009);
  - mensagens malformadas: resposta "erro" e a conexão continua;
  - streaming (ids "STREAM_"): só o quadro MAIS RECENTE fica pendente (os
    anteriores são descartados), pois o app envia quadros sem esperar resposta;
  - fotos: fila limitada por conexão; se cheia, responde "erro".
  - uma única inferência por vez no Pi (InferenceWorker), com prioridade ao app.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import websockets
from websockets.asyncio.server import Server, ServerConnection, serve

from .logging_setup import get_logger, log_event
from .models import UNKNOWN_PACKET_ID, Detection, InferenceResult, StatusResponse
from .protocol import ImagePacket, PingMessage, ProtocolError, parse_frame, parse_text, pong_message

log = get_logger("ws_server")

T = TypeVar("T")
MAX_CONNECTIONS = 4
ERROR_PREFIX = "Erro no processamento pelo servidor: "


class InferenceWorker:
    """Serializa as inferências em uma única thread; pedidos do app passam na frente."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self._lock = asyncio.Lock()
        self._app_waiting = 0

    async def run_app(self, fn: Callable[..., T], *args: Any) -> T:
        self._app_waiting += 1
        try:
            async with self._lock:
                return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)
        finally:
            self._app_waiting -= 1

    async def run_camera(self, fn: Callable[..., T], *args: Any) -> T:
        while self._app_waiting:
            await asyncio.sleep(0.05)
        async with self._lock:
            return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


AlertCallback = Callable[[ImagePacket, list[Detection]], Awaitable[None]]


class _Connection:
    def __init__(self, ws: ServerConnection, max_photos: int) -> None:
        self.ws = ws
        self.peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "?"
        self.stream_slot: ImagePacket | None = None
        self.stream_ready = asyncio.Event()
        self.photos: asyncio.Queue[ImagePacket] = asyncio.Queue(maxsize=max_photos)
        self.stream_dropped = 0


class AppGateway:
    def __init__(self, detect_jpeg: Callable[[bytes], InferenceResult], worker: InferenceWorker,
                 host: str = "0.0.0.0", port: int = 8000, path: str = "/ws",
                 max_frame_bytes: int = 5 * 1024 * 1024, max_id_bytes: int = 256, max_pending_photos: int = 4,
                 on_alert: AlertCallback | None = None) -> None:
        self.detect_jpeg = detect_jpeg
        self.worker = worker
        self.host, self.port, self.path = host, port, path
        self.max_frame_bytes, self.max_id_bytes = max_frame_bytes, max_id_bytes
        self.max_pending_photos = max_pending_photos
        self.on_alert = on_alert
        self._server: Server | None = None
        self.connections: set[_Connection] = set()
        self.processed = 0
        self.rejected = 0

    async def start(self) -> None:
        self._server = await serve(self._handler, self.host, self.port, max_size=self.max_frame_bytes,
                                   max_queue=4, ping_interval=20, ping_timeout=20, close_timeout=5)
        self.port = self._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        log_event(log, "app_ws_listening", url=f"ws://{self.host}:{self.port}{self.path}",
                  max_frame_bytes=self.max_frame_bytes)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    # ------------------------------------------------------------ conexão
    async def _handler(self, ws: ServerConnection) -> None:
        if ws.request is not None and ws.request.path != self.path:
            await ws.close(1008, "caminho inválido; use " + self.path)
            return
        if len(self.connections) >= MAX_CONNECTIONS:
            await ws.close(1013, "muitas conexões")
            return
        conn = _Connection(ws, self.max_pending_photos)
        self.connections.add(conn)
        log_event(log, "app_connected", peer=conn.peer, active=len(self.connections))
        tasks = [asyncio.create_task(self._stream_loop(conn)), asyncio.create_task(self._photo_loop(conn))]
        try:
            async for message in ws:
                await self._on_message(conn, message)
        except websockets.ConnectionClosed as exc:
            log_event(log, "app_connection_closed", level=30, peer=conn.peer, code=exc.rcvd.code if exc.rcvd else None)
        finally:
            for t in tasks:
                t.cancel()
            self.connections.discard(conn)
            log_event(log, "app_disconnected", peer=conn.peer, active=len(self.connections),
                      stream_dropped=conn.stream_dropped)

    async def _on_message(self, conn: _Connection, message: str | bytes) -> None:
        try:
            parsed = (parse_text(message, self.max_id_bytes) if isinstance(message, str)
                      else parse_frame(message, self.max_id_bytes))
        except ProtocolError as exc:
            self.rejected += 1
            log_event(log, "app_bad_message", level=30, peer=conn.peer, error=str(exc), size=len(message))
            await self._send(conn, StatusResponse.error(exc.packet_id or UNKNOWN_PACKET_ID, ERROR_PREFIX + str(exc)))
            return
        if isinstance(parsed, PingMessage):
            await conn.ws.send(pong_message())
            return
        if parsed.is_stream:
            if conn.stream_slot is not None:
                conn.stream_dropped += 1  # descarta o quadro mais antigo
            conn.stream_slot = parsed
            conn.stream_ready.set()
            return
        try:
            conn.photos.put_nowait(parsed)
        except asyncio.QueueFull:
            self.rejected += 1
            await self._send(conn, StatusResponse.error(parsed.packet_id, ERROR_PREFIX + "fila do nó cheia, tente novamente"))
            return
        await self._send(conn, StatusResponse.analyzing(parsed.packet_id))

    # ------------------------------------------------------------ processamento
    async def _stream_loop(self, conn: _Connection) -> None:
        while True:
            await conn.stream_ready.wait()
            conn.stream_ready.clear()
            packet, conn.stream_slot = conn.stream_slot, None
            if packet is not None:
                await self._process(conn, packet)

    async def _photo_loop(self, conn: _Connection) -> None:
        while True:
            packet = await conn.photos.get()
            await self._process(conn, packet)

    async def _process(self, conn: _Connection, packet: ImagePacket) -> None:
        try:
            result = await self.worker.run_app(self.detect_jpeg, packet.jpeg)
        except ValueError as exc:  # JPEG que não decodifica (mesma mensagem do servidor)
            await self._send(conn, StatusResponse.error(packet.packet_id, ERROR_PREFIX + str(exc)))
            return
        except Exception as exc:
            log.exception("falha na inferência do pacote %s", packet.packet_id)
            await self._send(conn, StatusResponse.error(packet.packet_id, ERROR_PREFIX + str(exc)))
            return
        self.processed += 1
        log_event(log, "app_inference", peer=conn.peer, packet_id=packet.packet_id, stream=packet.is_stream,
                  detections=len(result.detections), infer_ms=round(result.inference_ms, 1),
                  total_ms=round(result.total_ms, 1))
        await self._send(conn, StatusResponse.processed(packet.packet_id, result.detections))
        if not packet.is_stream and self.on_alert is not None and result.detections:
            try:
                await self.on_alert(packet, result.detections)
            except Exception:
                log.exception("falha ao encaminhar alerta do app")

    async def _send(self, conn: _Connection, response: StatusResponse) -> None:
        try:
            await conn.ws.send(response.to_json())
        except websockets.ConnectionClosed:
            pass
