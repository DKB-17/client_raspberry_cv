"""Descoberta do servidor por QR Code.

Contrato (servidor_cv/src/main.py): o servidor imprime no terminal um QR cujo
conteúdo é texto puro "ws://<ip_local>:<SERVER_PORT>/ws". A normalização abaixo
replica a do app (app_flutter_cv/.../config_page.dart):
  - aceita ws://, wss://, http:// (vira ws://) e https:// (vira wss://);
  - acrescenta "/ws" ao final se faltar.
O servidor também envia no QR o intervalo de captura dos nós, como query string:
  ws://<ip>:8000/ws?capture_interval=5   (segundos entre quadros analisados)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit, urlunsplit

import cv2
import numpy as np

from .logging_setup import get_logger, log_event

log = get_logger("discovery")

MAX_URL_LEN = 512
CAPTURE_INTERVAL_PARAM = "capture_interval"
MIN_CAPTURE_INTERVAL_S = 0.5  # mesma faixa do servidor (CAPTURE_INTERVAL_S)
MAX_CAPTURE_INTERVAL_S = 3600.0
_SCHEMES = {"ws": "ws", "wss": "wss", "http": "ws", "https": "wss"}


class InvalidServerUrl(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ServerInfo:
    url: str  # ws(s)://host:porta/ws, sem query string
    capture_interval_s: float | None  # None se o QR não trouxer o intervalo


def parse_server_qr(text: str) -> ServerInfo:
    """Valida o conteúdo do QR e separa a URL do intervalo de captura."""
    url = normalize_server_url(text)
    query = parse_qs(urlsplit(text.strip()).query, keep_blank_values=True)
    values = query.get(CAPTURE_INTERVAL_PARAM)
    if not values:
        return ServerInfo(url, None)
    try:
        interval = float(values[-1])
    except ValueError as exc:
        raise InvalidServerUrl(f"{CAPTURE_INTERVAL_PARAM} inválido: {values[-1]!r}") from exc
    if not MIN_CAPTURE_INTERVAL_S <= interval <= MAX_CAPTURE_INTERVAL_S:
        raise InvalidServerUrl(f"{CAPTURE_INTERVAL_PARAM} fora da faixa "
                               f"{MIN_CAPTURE_INTERVAL_S:g}–{MAX_CAPTURE_INTERVAL_S:g} s: {interval:g}")
    return ServerInfo(url, interval)


def normalize_server_url(text: str) -> str:
    """Valida e normaliza o conteúdo lido do QR para ws(s)://host[:porta]/.../ws.

    A query string só pode conter `capture_interval` (lido por parse_server_qr) e
    é removida da URL de conexão.
    """
    if not isinstance(text, str):
        raise InvalidServerUrl("conteúdo não é texto")
    url = text.strip()
    if not url:
        raise InvalidServerUrl("URL vazia")
    if len(url) > MAX_URL_LEN:
        raise InvalidServerUrl(f"URL maior que {MAX_URL_LEN} caracteres")
    if any(ch.isspace() for ch in url):
        raise InvalidServerUrl("URL contém espaços")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in _SCHEMES:
        raise InvalidServerUrl(f"esquema não suportado: {parts.scheme or '(nenhum)'}; use ws:// ou wss://")
    if not parts.hostname:
        raise InvalidServerUrl("URL sem host")
    try:
        port = parts.port
    except ValueError as exc:
        raise InvalidServerUrl("porta inválida") from exc
    if port == 0:
        raise InvalidServerUrl("porta inválida")
    if parts.username or parts.password:
        raise InvalidServerUrl("credenciais na URL não são permitidas")
    if parts.fragment:
        raise InvalidServerUrl("URL não deve ter fragmento")
    unknown = set(parse_qs(parts.query, keep_blank_values=True)) - {CAPTURE_INTERVAL_PARAM}
    if unknown:
        raise InvalidServerUrl(f"parâmetro(s) desconhecido(s) na URL: {', '.join(sorted(unknown))}")
    path = parts.path or ""
    if not path.endswith("/ws"):
        path = path + ("ws" if path.endswith("/") else "/ws")
    return urlunsplit((_SCHEMES[scheme], parts.netloc, path, "", ""))


def http_base_url(ws_url: str) -> str:
    """ws://h:p/ws -> http://h:p (usado no selftest para GET /)."""
    parts = urlsplit(ws_url)
    scheme = "https" if parts.scheme == "wss" else "http"
    return urlunsplit((scheme, parts.netloc, "", "", ""))


_detector: cv2.QRCodeDetector | None = None


QR_MIN_SCALED_WIDTH = 640  # abaixo disso não reduz: o QR a ~1 m ficaria pequeno demais


def decode_qr(frame_bgr: np.ndarray, scale: float = 1.0) -> str | None:
    """Tenta ler um QR no quadro. Retorna o texto ou None.

    `scale` < 1 reduz o quadro antes de decodificar (camera.qr_scale), o que
    corta bastante o tempo no Pi 3; nunca reduz abaixo de QR_MIN_SCALED_WIDTH.
    """
    global _detector
    if _detector is None:
        _detector = cv2.QRCodeDetector()
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    scale = max(scale, min(1.0, QR_MIN_SCALED_WIDTH / gray.shape[1]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    try:
        text, points, _ = _detector.detectAndDecode(gray)
    except cv2.error:
        return None
    return text or None


def scan_for_server_url(capture: Callable[[], np.ndarray], timeout_s: float = 120.0,
                        interval_s: float = 0.05, scale: float = 1.0) -> ServerInfo:
    """Lê quadros até achar um QR com URL válida ou estourar o tempo.

    QRs com conteúdo inválido são registrados e ignorados (continua procurando).
    """
    deadline = time.monotonic() + timeout_s
    last_invalid: str | None = None
    while time.monotonic() < deadline:
        text = decode_qr(capture(), scale)
        if text:
            try:
                info = parse_server_qr(text)
                log_event(log, "qr_found", url=info.url, capture_interval_s=info.capture_interval_s)
                return info
            except InvalidServerUrl as exc:
                if text != last_invalid:
                    log_event(log, "qr_invalid", level=30, content=text[:80], reason=str(exc))
                    last_invalid = text
        if interval_s > 0:
            time.sleep(interval_s)
    raise TimeoutError(f"nenhum QR Code válido lido em {timeout_s:.0f} s")
