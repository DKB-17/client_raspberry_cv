"""Protocolo binário do WebSocket /ws (app <-> servidor, app <-> nó, nó -> servidor).

Fonte da verdade:
  - app_flutter_cv/.../websocket_service.dart (sendPacket / sendStreamFrame)
  - servidor_cv/src/application/message_handler.py

Quadro binário:  [uint32 Big-Endian: N] [N bytes: id em UTF-8] [bytes do JPEG]
Texto (ping):    {"type": "ping"}  ->  {"type": "pong", "connection_id": "<uuid4>"}
Texto (legado):  {"id": "...", "image_data": "<base64>", "timestamp": "..."}
"""

from __future__ import annotations

import base64
import binascii
import json
import struct
import uuid
from dataclasses import dataclass

from .models import STREAM_PREFIX

HEADER = struct.Struct(">I")
JPEG_MAGIC = b"\xff\xd8"


class ProtocolError(ValueError):
    """Mensagem malformada. `packet_id` é preenchido quando já foi possível lê-lo."""

    def __init__(self, message: str, packet_id: str | None = None) -> None:
        super().__init__(message)
        self.packet_id = packet_id


@dataclass(frozen=True, slots=True)
class ImagePacket:
    packet_id: str
    jpeg: bytes

    @property
    def is_stream(self) -> bool:
        return self.packet_id.startswith(STREAM_PREFIX)


@dataclass(frozen=True, slots=True)
class PingMessage:
    pass


def pack_frame(packet_id: str, jpeg: bytes) -> bytes:
    id_bytes = packet_id.encode("utf-8")
    return HEADER.pack(len(id_bytes)) + id_bytes + jpeg


def new_packet_id() -> str:
    return str(uuid.uuid4())


def new_stream_id() -> str:
    # mesmo formato do app: "STREAM_" + 8 primeiros caracteres de um UUID v4
    return STREAM_PREFIX + uuid.uuid4().hex[:8]


def _check_image(packet_id: str, jpeg: bytes) -> None:
    if not jpeg:
        raise ProtocolError("imagem vazia", packet_id)
    if not jpeg.startswith(JPEG_MAGIC):
        raise ProtocolError("conteúdo não é JPEG (assinatura FF D8 ausente)", packet_id)


def parse_frame(data: bytes, max_id_bytes: int = 256) -> ImagePacket:
    """Valida e separa id e JPEG de um quadro binário."""
    if len(data) < HEADER.size:
        raise ProtocolError(f"quadro com {len(data)} bytes; cabeçalho exige 4")
    (id_len,) = HEADER.unpack_from(data)
    if id_len == 0:
        raise ProtocolError("tamanho do id é zero")
    if id_len > max_id_bytes:
        raise ProtocolError(f"tamanho do id ({id_len}) excede o máximo ({max_id_bytes})")
    if len(data) < HEADER.size + id_len:
        raise ProtocolError(f"quadro truncado: id declara {id_len} bytes, há {len(data) - 4}")
    try:
        packet_id = data[HEADER.size:HEADER.size + id_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("id não é UTF-8 válido") from exc
    if not packet_id.isprintable():
        raise ProtocolError("id contém caracteres de controle")
    jpeg = bytes(data[HEADER.size + id_len:])
    _check_image(packet_id, jpeg)
    return ImagePacket(packet_id, jpeg)


def parse_text(text: str, max_id_bytes: int = 256) -> PingMessage | ImagePacket:
    """Mensagem de texto: ping ou pacote JSON com imagem em base64 (fallback do servidor)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError("texto não é JSON válido") from exc
    if not isinstance(data, dict):
        raise ProtocolError("JSON deve ser um objeto")
    if data.get("type") == "ping":
        return PingMessage()
    packet_id = data.get("id")
    image_data = data.get("image_data")
    if not isinstance(packet_id, str) or not packet_id:
        raise ProtocolError("campo 'id' ausente ou inválido")
    if len(packet_id.encode("utf-8")) > max_id_bytes or not packet_id.isprintable():
        raise ProtocolError("campo 'id' inválido", None)
    if not isinstance(image_data, str):
        raise ProtocolError("campo 'image_data' ausente", packet_id)
    try:
        jpeg = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ProtocolError("'image_data' não é base64 válido", packet_id) from exc
    _check_image(packet_id, jpeg)
    return ImagePacket(packet_id, jpeg)


def pong_message() -> str:
    return json.dumps({"type": "pong", "connection_id": str(uuid.uuid4())})


def ping_message() -> str:
    return json.dumps({"type": "ping"})
