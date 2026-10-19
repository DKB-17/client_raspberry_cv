"""Fila persistente (SQLite) de alertas a entregar ao servidor central.

Sobrevive a quedas de rede e reinícios. Limitada por quantidade e por bytes:
ao estourar, descarta os itens mais antigos com aviso no log.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .logging_setup import get_logger, log_event

log = get_logger("outbox")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    packet_id       TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    next_attempt_at REAL NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    origin          TEXT NOT NULL,
    detections      TEXT NOT NULL,
    size            INTEGER NOT NULL,
    jpeg            BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_due ON outbox (next_attempt_at, created_at);
"""


@dataclass(frozen=True, slots=True)
class OutboxItem:
    packet_id: str
    created_at: float
    attempts: int
    origin: str
    detections: str
    jpeg: bytes


class Outbox:
    def __init__(self, path: Path, max_items: int = 200, max_bytes: int = 100 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_items, self.max_bytes = max_items, max_bytes
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")  # menos escrita no cartão SD
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def put(self, packet_id: str, jpeg: bytes, detections_json: str, origin: str,
            now: float | None = None) -> list[str]:
        """Enfileira; retorna ids descartados por falta de espaço."""
        now = time.time() if now is None else now
        if len(jpeg) > self.max_bytes:
            raise ValueError("imagem maior que o limite total da fila")
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO outbox (packet_id, created_at, next_attempt_at, origin, detections,"
                " size, jpeg) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (packet_id, now, now, origin, detections_json, len(jpeg), jpeg))
            dropped = self._evict_locked()
        for pid in dropped:
            log_event(log, "outbox_dropped_oldest", level=30, packet_id=pid,
                      max_items=self.max_items, max_bytes=self.max_bytes)
        return dropped

    def _evict_locked(self) -> list[str]:
        dropped: list[str] = []
        while True:
            count, total = self._db.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM outbox").fetchone()
            if count <= self.max_items and total <= self.max_bytes:
                return dropped
            (pid,) = self._db.execute("SELECT packet_id FROM outbox ORDER BY created_at LIMIT 1").fetchone()
            self._db.execute("DELETE FROM outbox WHERE packet_id = ?", (pid,))
            dropped.append(pid)

    def next_due(self, now: float | None = None) -> OutboxItem | None:
        now = time.time() if now is None else now
        with self._lock:
            row = self._db.execute(
                "SELECT packet_id, created_at, attempts, origin, detections, jpeg FROM outbox"
                " WHERE next_attempt_at <= ? ORDER BY created_at LIMIT 1", (now,)).fetchone()
        return OutboxItem(*row) if row else None

    def seconds_until_next(self, now: float | None = None) -> float | None:
        now = time.time() if now is None else now
        with self._lock:
            row = self._db.execute("SELECT MIN(next_attempt_at) FROM outbox").fetchone()
        return None if row[0] is None else max(0.0, row[0] - now)

    def mark_failed(self, packet_id: str, retry_at: float, error: str) -> int:
        with self._lock:
            self._db.execute("UPDATE outbox SET attempts = attempts + 1, next_attempt_at = ?, last_error = ?"
                             " WHERE packet_id = ?", (retry_at, error[:500], packet_id))
            row = self._db.execute("SELECT attempts FROM outbox WHERE packet_id = ?", (packet_id,)).fetchone()
        return row[0] if row else 0

    def delete(self, packet_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM outbox WHERE packet_id = ?", (packet_id,))

    def stats(self) -> tuple[int, int]:
        with self._lock:
            count, total = self._db.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM outbox").fetchone()
        return int(count), int(total)
