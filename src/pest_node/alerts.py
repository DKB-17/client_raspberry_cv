"""Regras de alerta.

Uma detecção dispara alerta quando a confiança >= threshold e a classe não
disparou nos últimos `cooldown_s` segundos (evita inundar o servidor com a
mesma praga a cada quadro). O alerta em si é enviar a imagem ao servidor
central, que a persiste em data/<data>/classified e no PostgreSQL.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .models import Detection


@dataclass
class AlertPolicy:
    threshold: float
    cooldown_s: float = 60.0
    classes: list[str] = field(default_factory=list)
    _last: dict[str, float] = field(default_factory=dict, init=False, repr=False)

    def triggering(self, detections: list[Detection], now: float | None = None) -> list[Detection]:
        """Retorna as detecções que disparam alerta e registra o disparo por classe."""
        now = time.monotonic() if now is None else now
        hits: dict[str, Detection] = {}
        for det in detections:
            if det.confidence < self.threshold:
                continue
            if self.classes and det.class_name not in self.classes:
                continue
            last = self._last.get(det.class_name)
            if last is not None and now - last < self.cooldown_s:
                continue
            best = hits.get(det.class_name)
            if best is None or det.confidence > best.confidence:
                hits[det.class_name] = det
        for name in hits:
            self._last[name] = now
        return list(hits.values())
