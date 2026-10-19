"""Leitura de temperatura e throttling (apenas para logs e autoteste; não altera a coleta).

`vcgencmd get_throttled` devolve um bitmask (documentação Raspberry Pi):
  bit 0  subtensão agora             bit 16 houve subtensão
  bit 1  frequência limitada agora   bit 17 houve limitação de frequência
  bit 2  throttling agora            bit 18 houve throttling
  bit 3  limite térmico suave agora  bit 19 houve limite térmico suave
Subtensão (fonte fraca) e temperatura derrubam a CPU do mesmo jeito, mas têm
causas diferentes; por isso cada bit é registrado separadamente.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ThrottleFlags:
    raw: int

    def _bit(self, n: int) -> bool:
        return bool(self.raw >> n & 1)

    under_voltage_now = property(lambda self: self._bit(0))
    freq_capped_now = property(lambda self: self._bit(1))
    throttled_now = property(lambda self: self._bit(2))
    soft_temp_limit_now = property(lambda self: self._bit(3))
    under_voltage_occurred = property(lambda self: self._bit(16))
    freq_capped_occurred = property(lambda self: self._bit(17))
    throttled_occurred = property(lambda self: self._bit(18))
    soft_temp_limit_occurred = property(lambda self: self._bit(19))

    def active(self) -> list[str]:
        names = {0: "under_voltage", 1: "freq_capped", 2: "throttled", 3: "soft_temp_limit"}
        return [n for b, n in names.items() if self._bit(b)]

    def describe(self) -> str:
        now = self.active()
        past = [n for b, n in {16: "under_voltage", 17: "freq_capped", 18: "throttled",
                               19: "soft_temp_limit"}.items() if self._bit(b)]
        return f"0x{self.raw:x} agora={now or '-'} desde_boot={past or '-'}"


@dataclass(frozen=True, slots=True)
class ThermalSample:
    timestamp: float
    temp_c: float | None
    flags: ThrottleFlags | None
    arm_freq_mhz: float | None = None


class ThermalSensor(Protocol):
    def read(self) -> ThermalSample: ...


# --------------------------------------------------------------------------- #
#  Sensores
# --------------------------------------------------------------------------- #
_TEMP_RE = re.compile(r"temp=([\d.]+)")
_THROTTLE_RE = re.compile(r"throttled=(0x[0-9a-fA-F]+)")
_FREQ_RE = re.compile(r"=(\d+)")


def parse_measure_temp(text: str) -> float | None:
    m = _TEMP_RE.search(text)
    return float(m.group(1)) if m else None


def parse_get_throttled(text: str) -> int | None:
    m = _THROTTLE_RE.search(text)
    return int(m.group(1), 16) if m else None


def parse_measure_clock(text: str) -> float | None:
    m = _FREQ_RE.search(text)
    return int(m.group(1)) / 1e6 if m else None


class VcgencmdSensor:
    """Lê via `vcgencmd`; se ausente, usa /sys/class/thermal (sem bits de throttling)."""

    SYSFS_TEMP = Path("/sys/class/thermal/thermal_zone0/temp")

    def __init__(self, timeout_s: float = 2.0) -> None:
        self.vcgencmd = shutil.which("vcgencmd")
        self.timeout_s = timeout_s

    @property
    def available(self) -> bool:
        return self.vcgencmd is not None or self.SYSFS_TEMP.exists()

    def _run(self, *args: str) -> str | None:
        if not self.vcgencmd:
            return None
        try:
            res = subprocess.run([self.vcgencmd, *args], capture_output=True, text=True,
                                 timeout=self.timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return res.stdout if res.returncode == 0 else None

    def _sysfs_temp(self) -> float | None:
        try:
            return int(self.SYSFS_TEMP.read_text().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def read(self) -> ThermalSample:
        temp_txt = self._run("measure_temp")
        temp = parse_measure_temp(temp_txt) if temp_txt else self._sysfs_temp()
        thr_txt = self._run("get_throttled")
        raw = parse_get_throttled(thr_txt) if thr_txt else None
        clk_txt = self._run("measure_clock", "arm")
        freq = parse_measure_clock(clk_txt) if clk_txt else None
        return ThermalSample(time.time(), temp, ThrottleFlags(raw) if raw is not None else None, freq)


class SimulatedSensor:
    """Sensor falso para testes e para rodar no PC: repete uma sequência de leituras."""

    def __init__(self, readings: list[tuple[float | None, int | None]] | None = None,
                 loop: bool = True) -> None:
        self.readings = readings or [(45.0, 0)]
        self.loop = loop
        self._it: Iterator[tuple[float | None, int | None]] = iter(self.readings)
        self.available = True

    def read(self) -> ThermalSample:
        try:
            temp, raw = next(self._it)
        except StopIteration:
            if not self.loop:
                temp, raw = self.readings[-1]
            else:
                self._it = iter(self.readings)
                temp, raw = next(self._it)
        return ThermalSample(time.time(), temp, ThrottleFlags(raw) if raw is not None else None, None)
