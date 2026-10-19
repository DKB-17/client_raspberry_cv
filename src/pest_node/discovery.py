"""Descoberta do servidor por QR Code.

Contrato (servidor_cv/src/main.py): o servidor imprime no terminal um QR cujo
conteúdo é texto puro "ws://<ip_local>:<SERVER_PORT>/ws". A normalização abaixo
replica a do app (app_flutter_cv/.../config_page.dart):
  - aceita ws://, wss://, http:// (vira ws://) e https:// (vira wss://);
  - acrescenta "/ws" ao final se faltar.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from .logging_setup import get_logger, log_event

log = get_logger("discovery")

MAX_URL_LEN = 512
_SCHEMES = {"ws": "ws", "wss": "wss", "http": "ws", "https": "wss"}


class InvalidServerUrl(ValueError):
    pass


def normalize_server_url(text: str) -> str:
    """Valida e normaliza o conteúdo lido do QR para ws(s)://host[:porta]/.../ws."""
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
    if parts.query or parts.fragment:
        raise InvalidServerUrl("URL não deve ter query string nem fragmento")
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


def decode_qr(frame_bgr: np.ndarray) -> str | None:
    """Tenta ler um QR no quadro. Retorna o texto ou None."""
    global _detector
    if _detector is None:
        _detector = cv2.QRCodeDetector()
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    try:
        text, points, _ = _detector.detectAndDecode(gray)
    except cv2.error:
        return None
    return text or None


def scan_for_server_url(capture: Callable[[], np.ndarray], timeout_s: float = 120.0,
                        interval_s: float = 0.3) -> str:
    """Lê quadros até achar um QR com URL válida ou estourar o tempo.

    QRs com conteúdo inválido são registrados e ignorados (continua procurando).
    """
    deadline = time.monotonic() + timeout_s
    last_invalid: str | None = None
    while time.monotonic() < deadline:
        text = decode_qr(capture())
        if text:
            try:
                url = normalize_server_url(text)
                log_event(log, "qr_found", url=url)
                return url
            except InvalidServerUrl as exc:
                if text != last_invalid:
                    log_event(log, "qr_invalid", level=30, content=text[:80], reason=str(exc))
                    last_invalid = text
        time.sleep(interval_s)
    raise TimeoutError(f"nenhum QR Code válido lido em {timeout_s:.0f} s")
