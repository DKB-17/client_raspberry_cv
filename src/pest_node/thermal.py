"""Temperatura, throttling e controle adaptativo de carga.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .logging_setup import get_logger, log_event

log = get_logger("thermal")


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


# --------------------------------------------------------------------------- #
#  Controle adaptativo
# --------------------------------------------------------------------------- #
@dataclass
class AdaptiveController:
    """Ajusta o intervalo entre inferências conforme temperatura e throttling.

    - "Pressão" (temp >= temp_high_c, throttling/limite térmico ou subtensão,
      conforme configuração): multiplica o intervalo por slowdown_factor a cada
      leitura, até max_interval_s.
    - Alívio (temp <= temp_resume_c e nenhum bit ativo): divide pelo mesmo fator
      a cada leitura até voltar ao intervalo base. Entre os dois limiares, mantém
      (histerese).
    Aumentar o intervalo equivale a pular quadros: a câmera continua rodando,
    apenas menos quadros são inferidos.
    """

    target_fps: float
    temp_high_c: float = 70.0
    temp_resume_c: float = 65.0
    slowdown_factor: float = 2.0
    max_interval_s: float = 30.0
    react_to_throttling: bool = True
    react_to_under_voltage: bool = True
    interval_s: float = field(init=False)
    reason: str = field(init=False, default="normal")
    last_sample: ThermalSample | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.base_interval_s = 1.0 / self.target_fps
        self.interval_s = self.base_interval_s

    @property
    def degraded(self) -> bool:
        return self.interval_s > self.base_interval_s * 1.0001

    def _pressure(self, s: ThermalSample) -> list[str]:
        causes: list[str] = []
        if s.temp_c is not None and s.temp_c >= self.temp_high_c:
            causes.append(f"temp>={self.temp_high_c:g}C")
        if s.flags is not None:
            if self.react_to_throttling and (s.flags.throttled_now or s.flags.soft_temp_limit_now
                                             or s.flags.freq_capped_now):
                causes.append("throttling")
            if self.react_to_under_voltage and s.flags.under_voltage_now:
                causes.append("subtensao")
        return causes

    def _relieved(self, s: ThermalSample) -> bool:
        cool = s.temp_c is None or s.temp_c <= self.temp_resume_c
        return cool and not self._pressure(s)

    def update(self, sample: ThermalSample) -> float:
        self.last_sample = sample
        previous = self.interval_s
        causes = self._pressure(sample)
        if causes:
            self.interval_s = min(self.interval_s * self.slowdown_factor, self.max_interval_s)
            self.reason = "+".join(causes)
        elif self._relieved(sample) and self.degraded:
            self.interval_s = max(self.interval_s / self.slowdown_factor, self.base_interval_s)
            self.reason = "normal" if not self.degraded else "resfriando"
        if self.interval_s != previous:
            log_event(log, "load_adjusted", level=30 if self.interval_s > previous else 20,
                      interval_s=round(self.interval_s, 2), previous_s=round(previous, 2), reason=self.reason,
                      temp_c=sample.temp_c, throttled=sample.flags.describe() if sample.flags else None)
        return self.interval_s
