"""Uso de CPU e memória lidos de /proc (Linux), com fallback para psutil no PC."""

from __future__ import annotations

import os
from pathlib import Path

try:  # opcional, só para desenvolvimento fora do Linux
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    psutil = None

_PROC = Path("/proc")


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def process_rss_mb() -> float | None:
    status = _read(_PROC / "self" / "status")
    if status:
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    if psutil:
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    return None


def mem_available_mb() -> float | None:
    info = _read(_PROC / "meminfo")
    if info:
        for line in info.splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    if psutil:
        return psutil.virtual_memory().available / 1024 / 1024
    return None


class CpuMeter:
    """CPU total do sistema (%) entre duas chamadas consecutivas de percent()."""

    def __init__(self) -> None:
        self._last = self._times()
        if self._last is None and psutil:
            psutil.cpu_percent(None)

    @staticmethod
    def _times() -> tuple[int, int] | None:
        stat = _read(_PROC / "stat")
        if not stat:
            return None
        values = [int(v) for v in stat.splitlines()[0].split()[1:]]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return idle, sum(values)

    def percent(self) -> float | None:
        now = self._times()
        if now is None:
            return psutil.cpu_percent(None) if psutil else None
        if self._last is None:
            self._last = now
            return None
        idle = now[0] - self._last[0]
        total = now[1] - self._last[1]
        self._last = now
        return 100.0 * (1 - idle / total) if total > 0 else None
