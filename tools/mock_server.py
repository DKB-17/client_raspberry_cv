"""Servidor de teste que reproduz e VALIDA o contrato do servidor_cv (/ws).

Comportamento copiado de servidor_cv/src/presentation/websocket_router.py e
src/application/message_handler.py:
  - texto {"type":"ping"} -> {"type":"pong","connection_id":"<uuid>"}
  - binário [u32 BE][id UTF-8][JPEG]: se id não começa com "STREAM_", envia
    "analisando"; depois "processado" com detecções (ou "erro").
  - a imagem precisa decodificar (cv2.imdecode), senão "erro" com a mesma
    mensagem do servidor.
Também registra cada pacote recebido, para os testes conferirem o contrato.

Uso manual:  python tools/mock_server.py --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from websockets.asyncio.server import Server, ServerConnection, serve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pest_node.models import Detection, StatusResponse  # noqa: E402

DEFAULT_DETECTION = Detection(7, "Percevejo-verde-ninfa", 0.9142, (152.7, 428.1, 318.7, 565.3))


@dataclass
class ReceivedPacket:
    packet_id: str
    jpeg_size: int
    is_stream: bool
    image_shape: tuple[int, ...] | None


@dataclass
class MockServer:
    host: str = "127.0.0.1"
    port: int = 0
    detections: Callable[[str], list[Detection]] = lambda _id: [DEFAULT_DETECTION]
    delay_s: float = 0.0
    error_ids: set[str] = field(default_factory=set)  # responde "erro" para esses ids
    drop_ids: set[str] = field(default_factory=set)  # fecha a conexão sem responder (uma vez)
    received: list[ReceivedPacket] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    pings: int = 0
    _server: Server | None = None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws"

    async def start(self) -> MockServer:
        self._server = await serve(self._handler, self.host, self.port, max_size=None)
        self.port = self._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return self

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> MockServer:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def _handler(self, ws: ServerConnection) -> None:
        if ws.request is not None and ws.request.path != "/ws":
            self.violations.append(f"caminho inválido: {ws.request.path}")
            await ws.close(1008)
            return
        async for message in ws:
            if isinstance(message, str):
                await self._on_text(ws, message)
            else:
                await self._on_binary(ws, message)

    async def _on_text(self, ws: ServerConnection, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            self.violations.append("texto não-JSON")
            await ws.send(StatusResponse.error("desconhecido", "Erro no processamento pelo servidor").to_json())
            return
        if isinstance(data, dict) and data.get("type") == "ping":
            self.pings += 1
            await ws.send(json.dumps({"type": "pong", "connection_id": str(uuid.uuid4())}))
            return
        self.violations.append(f"texto inesperado: {text[:80]}")

    async def _on_binary(self, ws: ServerConnection, raw: bytes) -> None:
        # Exatamente o parsing do servidor real:
        id_len = int.from_bytes(raw[:4], byteorder="big")
        try:
            packet_id = raw[4:4 + id_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            self.violations.append("id não UTF-8")
            await ws.send(StatusResponse.error("desconhecido", f"Erro no processamento pelo servidor: {exc}").to_json())
            return
        image_bytes = raw[4 + id_len:]
        is_stream = packet_id.startswith("STREAM_")
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR) if image_bytes else None
        self.received.append(ReceivedPacket(packet_id, len(image_bytes), is_stream,
                                            None if img is None else img.shape))
        if packet_id in self.drop_ids:
            self.drop_ids.discard(packet_id)
            await ws.close()
            return
        if not is_stream:
            await ws.send(StatusResponse.analyzing(packet_id).to_json())
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if img is None:
            self.violations.append(f"imagem não decodificável: {packet_id}")
            await ws.send(StatusResponse.error(packet_id, "Erro no processamento pelo servidor: "
                                               "Não foi possível decodificar a imagem recebida.").to_json())
            return
        if packet_id in self.error_ids:
            await ws.send(StatusResponse.error(packet_id, "Erro no processamento pelo servidor: simulado").to_json())
            return
        await ws.send(StatusResponse.processed(packet_id, self.detections(packet_id)).to_json())


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Servidor mock do contrato /ws do servidor_cv")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--delay", type=float, default=0.5, help="atraso simulado de inferência (s)")
    args = parser.parse_args()
    server = MockServer(args.host, args.port, delay_s=args.delay)
    await server.start()
    print(f"mock_server ouvindo em ws://{args.host}:{server.port}/ws (Ctrl+C para sair)")
    last = 0
    try:
        while True:
            await asyncio.sleep(1)
            for pkt in server.received[last:]:
                print(f"recebido id={pkt.packet_id} jpeg={pkt.jpeg_size}B stream={pkt.is_stream} "
                      f"imagem={pkt.image_shape}")
            last = len(server.received)
            for v in server.violations:
                print(f"VIOLAÇÃO DO CONTRATO: {v}")
            server.violations.clear()
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
