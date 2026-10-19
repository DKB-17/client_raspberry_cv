"""Logging estruturado com rotação.

- Arquivo: uma linha JSON por evento, com rotação por tamanho (não enche o SD).
- Console (stdout): texto legível; no Pi o journald captura essa saída.

Uso: log_event(logger, "inference", infer_ms=812.3, detections=2)
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any

ROOT_LOGGER = "pest_node"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = getattr(record, "fields", None)
        if extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        return base


def setup_logging(level: str = "INFO", log_dir: Path | None = None,
                  max_bytes: int = 2 * 1024 * 1024, backup_count: int = 3) -> logging.Logger:
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(level.upper())
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ConsoleFormatter())
    logger.addHandler(console)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_dir / "pest-node.log", maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT_LOGGER}.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": fields})
