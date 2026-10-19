import base64
import json
import struct

import pytest

from pest_node.models import Detection, StatusResponse
from pest_node.protocol import (PingMessage, ProtocolError, new_stream_id, pack_frame, parse_frame,
                                parse_text, pong_message)

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32 + b"\xff\xd9"
UUID = "48651cd1-32b5-42a3-8dbd-93aafee47985"


def test_pack_matches_flutter_layout():
    # websocket_service.dart: ByteData(4).setUint32(0, idLength, Endian.big) + id + imagem
    frame = pack_frame(UUID, JPEG)
    assert frame[:4] == struct.pack(">I", 36) == b"\x00\x00\x00\x24"
    assert frame[4:40] == UUID.encode("utf-8")
    assert frame[40:] == JPEG


def test_roundtrip_utf8_id():
    pid = "praga-açaí-ñ-🐛"
    pkt = parse_frame(pack_frame(pid, JPEG))
    assert pkt.packet_id == pid and pkt.jpeg == JPEG
    assert not pkt.is_stream


def test_stream_id_format():
    sid = new_stream_id()
    assert sid.startswith("STREAM_") and len(sid) == 15
    assert parse_frame(pack_frame(sid, JPEG)).is_stream


def test_server_parsing_equivalence():
    # mesma lógica de message_handler.py
    raw = pack_frame(UUID, JPEG)
    id_len = int.from_bytes(raw[:4], byteorder="big")
    assert raw[4:4 + id_len].decode("utf-8") == UUID
    assert raw[4 + id_len:] == JPEG


@pytest.mark.parametrize("data, msg", [
    (b"", "cabeçalho"),
    (b"\x00\x00", "cabeçalho"),
    (b"\x00\x00\x00\x00" + JPEG, "zero"),
    (struct.pack(">I", 10_000) + b"abc", "excede"),
    (struct.pack(">I", 40) + b"abc", "truncado"),
    (struct.pack(">I", 2) + b"\xc3\x28" + JPEG, "UTF-8"),
    (struct.pack(">I", 3) + b"a\x00b" + JPEG, "controle"),
    (struct.pack(">I", 3) + b"abc", "vazia"),
    (struct.pack(">I", 3) + b"abc" + b"GIF89a", "JPEG"),
    # little-endian por engano: 36 vira 603979776
    (struct.pack("<I", 36) + UUID.encode() + JPEG, "excede"),
])
def test_malformed_frames(data, msg):
    with pytest.raises(ProtocolError, match=msg):
        parse_frame(data)


def test_error_keeps_packet_id_when_known():
    with pytest.raises(ProtocolError) as exc:
        parse_frame(pack_frame("abc", b"nao-jpeg"))
    assert exc.value.packet_id == "abc"


def test_ping_and_pong():
    assert isinstance(parse_text('{"type":"ping"}'), PingMessage)
    pong = json.loads(pong_message())
    assert pong["type"] == "pong" and len(pong["connection_id"]) == 36


def test_text_fallback_base64():
    pkt = parse_text(json.dumps({"id": UUID, "image_data": base64.b64encode(JPEG).decode()}))
    assert pkt.packet_id == UUID and pkt.jpeg == JPEG


@pytest.mark.parametrize("text", ["nao json", "[1,2]", '{"id": 5}', '{"id":"a"}',
                                  '{"id":"a","image_data":"@@@"}'])
def test_text_malformed(text):
    with pytest.raises(ProtocolError):
        parse_text(text)


def test_status_response_matches_pydantic_dump():
    # Saída de RespostaStatusPacote(...).model_dump_json() do servidor (pydantic 2.7)
    det = Detection(7, "Percevejo-verde-ninfa", 0.914198, (152.71, 428.12, 318.66, 565.33))
    got = StatusResponse.processed(UUID, [det]).to_json()
    expected = ('{"id":"48651cd1-32b5-42a3-8dbd-93aafee47985","status":"processado",'
                '"mensagem":"1 detecção(ões) encontrada(s)","analysis":null,"detections":'
                '[{"class_name":"Percevejo-verde-ninfa","confidence":0.9142,"bbox":[152.7,428.1,318.7,565.3]}]}')
    assert got == expected
    assert StatusResponse.analyzing("x").to_json() == (
        '{"id":"x","status":"analisando","mensagem":"Imagem recebida e em análise",'
        '"analysis":null,"detections":null}')
