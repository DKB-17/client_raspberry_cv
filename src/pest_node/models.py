"""Estruturas do contrato com o servidor central e o app Flutter.

Fonte da verdade: servidor_cv/src/domain/communication.py
    DeteccaoResultado:   class_name: str, confidence: float, bbox: [x1, y1, x2, y2]
    RespostaStatusPacote: id, status ("analisando" | "processado" | "erro"),
                          mensagem, analysis, detections
O servidor serializa com Pydantic `model_dump_json()`: JSON compacto, sem
escapar acentos, com campos nulos presentes e nesta ordem de chaves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["analisando", "processado", "erro"]

STATUS_ANALYZING: Status = "analisando"
STATUS_PROCESSED: Status = "processado"
STATUS_ERROR: Status = "erro"

STREAM_PREFIX = "STREAM_"
UNKNOWN_PACKET_ID = "desconhecido"  # o servidor usa este id quando nem o id foi lido


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 em pixels da imagem original

    def to_wire(self) -> dict[str, Any]:
        """Formato do servidor (yolo_service.detect): conf 4 casas, bbox 1 casa."""
        return {
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
        }

    @classmethod
    def from_wire(cls, data: dict[str, Any], class_names: list[str] | None = None) -> Detection:
        name = str(data.get("class_name") or data.get("class") or "")
        bbox = data.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"bbox inválido: {bbox!r}")
        class_id = class_names.index(name) if class_names and name in class_names else -1
        return cls(class_id, name, float(data["confidence"]), tuple(float(v) for v in bbox))  # type: ignore[arg-type]


@dataclass(slots=True)
class StatusResponse:
    id: str
    status: Status
    mensagem: str | None = None
    analysis: str | None = None
    detections: list[Detection] | None = None

    def to_json(self) -> str:
        payload = {
            "id": self.id,
            "status": self.status,
            "mensagem": self.mensagem,
            "analysis": self.analysis,
            "detections": None if self.detections is None else [d.to_wire() for d in self.detections],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def analyzing(packet_id: str) -> StatusResponse:
        return StatusResponse(packet_id, STATUS_ANALYZING, "Imagem recebida e em análise")

    @staticmethod
    def processed(packet_id: str, detections: list[Detection]) -> StatusResponse:
        return StatusResponse(packet_id, STATUS_PROCESSED,
                              f"{len(detections)} detecção(ões) encontrada(s)", detections=detections)

    @staticmethod
    def error(packet_id: str, message: str) -> StatusResponse:
        return StatusResponse(packet_id, STATUS_ERROR, message)


@dataclass(slots=True)
class InferenceResult:
    detections: list[Detection]
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    source: Literal["local", "server"] = "local"

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(slots=True)
class AlertEvent:
    """Imagem com praga que será persistida pelo servidor (fluxo de alerta)."""

    packet_id: str
    created_at: float
    jpeg: bytes
    detections: list[Detection] = field(default_factory=list)
    origin: Literal["camera", "app"] = "camera"
