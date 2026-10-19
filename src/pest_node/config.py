"""Carga, validação e persistência do config.json.

Todas as opções do nó vivem em um único arquivo JSON. Valores inválidos geram
ConfigError com o caminho da chave (ex.: "thermal.temp_high_c"), nunca um
comportamento silencioso.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
import uuid
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config.json")
CONFIG_ENV_VAR = "PEST_NODE_CONFIG"


class ConfigError(ValueError):
    """Configuração inválida ou ilegível."""


@dataclass
class ModelConfig:
    path: str = "models/teste2.onnx"
    conf_threshold: float = 0.25  # mesmo padrão do servidor (YOLO_CONFIDENCE_THRESHOLD)
    iou_threshold: float = 0.7  # padrão do Ultralytics predict, usado pelo servidor
    max_detections: int = 300
    intra_op_num_threads: int = 2
    inter_op_num_threads: int = 1
    imgsz: int = 0  # 0 = lê do próprio ONNX; obrigatório só para modelos com entrada dinâmica
    class_names: list[str] = field(default_factory=list)  # vazio = nomes dos metadados do ONNX


@dataclass
class CameraConfig:
    width: int = 1296  # modo binado da ov5647 com campo de visão completo
    height: int = 972
    max_fps: float = 10.0  # limita o ISP; a inferência é bem mais lenta que isso
    jpeg_quality: int = 85
    fake_images_dir: str = ""  # se preenchido, usa a câmera falsa (pasta de imagens)


PROCESSING_MODES = ("auto", "local", "server")


@dataclass
class InferenceConfig:
    # De quantos em quantos segundos um quadro é capturado e analisado pelo YOLO.
    # Vem do QR Code do servidor (CAPTURE_INTERVAL_S); pode ser ajustado aqui.
    capture_interval_s: float = 5.0
    # Onde o YOLO roda:
    #   "auto"   no Pi; quando aquecido/estrangulado, envia o quadro ao servidor
    #   "local"  sempre no Pi
    #   "server" sempre no servidor (quadro STREAM_); se o servidor cair, roda no Pi
    mode: str = "auto"
    server_timeout_s: float = 15.0  # espera pela resposta do servidor ao enviar um quadro


@dataclass
class ThermalConfig:
    poll_interval_s: float = 5.0
    temp_high_c: float = 70.0
    temp_resume_c: float = 65.0
    react_to_throttling: bool = True
    react_to_under_voltage: bool = True
    slowdown_factor: float = 2.0
    max_interval_s: float = 30.0


@dataclass
class AlertConfig:
    threshold: float = 0.60
    cooldown_s: float = 60.0
    classes: list[str] = field(default_factory=list)  # vazio = todas as classes


@dataclass
class OutboxConfig:
    db_path: str = "data/outbox.sqlite"
    max_items: int = 200
    max_bytes: int = 100 * 1024 * 1024
    max_attempts: int = 5
    backoff_initial_s: float = 2.0
    backoff_max_s: float = 300.0
    response_timeout_s: float = 60.0


@dataclass
class WsServerConfig:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000  # mesma porta do servidor central; o app usa ws://<ip>:8000/ws
    path: str = "/ws"
    max_frame_bytes: int = 5 * 1024 * 1024
    max_id_bytes: int = 256
    max_pending_photos: int = 4
    forward_alerts: bool = True  # fotos do app com praga entram no fluxo de alerta


@dataclass
class LoggingConfig:
    level: str = "INFO"
    dir: str = "logs"
    max_bytes: int = 2 * 1024 * 1024
    backup_count: int = 3
    metrics_interval_s: float = 30.0


@dataclass
class Config:
    server_ws_url: str = ""  # gravado pelo scan-qr; ex.: ws://192.168.5.10:8000/ws
    node_id: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    outbox: OutboxConfig = field(default_factory=OutboxConfig)
    ws_server: WsServerConfig = field(default_factory=WsServerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Diretório do config.json; caminhos relativos são resolvidos a partir dele.
    base_dir: Path = field(default=Path("."), repr=False, compare=False)

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.base_dir / p)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("base_dir", None)
        return data


# --------------------------------------------------------------------------- #
#  Conversão dict -> dataclass com mensagens de erro claras
# --------------------------------------------------------------------------- #
def _coerce(value: Any, expected: Any, key: str) -> Any:
    if expected is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{key}: esperado true/false, recebido {value!r}")
        return value
    if expected is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{key}: esperado inteiro, recebido {value!r}")
        return value
    if expected is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{key}: esperado número, recebido {value!r}")
        return float(value)
    if expected is str:
        if not isinstance(value, str):
            raise ConfigError(f"{key}: esperado texto, recebido {value!r}")
        return value
    if expected == list[str]:
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{key}: esperado lista de textos, recebido {value!r}")
        return list(value)
    raise ConfigError(f"{key}: tipo não suportado")  # pragma: no cover


_TYPES = {"bool": bool, "int": int, "float": float, "str": str, "list[str]": list[str]}


def _build(cls: type, data: dict[str, Any], prefix: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"{prefix or 'raiz'}: esperado objeto JSON")
    known = {f.name: f for f in fields(cls) if f.name != "base_dir"}
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigError(f"{prefix or 'raiz'}: chave(s) desconhecida(s): {', '.join(sorted(unknown))}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        key = f"{prefix}.{name}" if prefix else name
        default = f.default_factory() if callable(f.default_factory) else f.default  # type: ignore[misc]
        if is_dataclass(default):
            kwargs[name] = _build(type(default), data[name], key)
        else:
            kwargs[name] = _coerce(data[name], _TYPES[str(f.type)], key)
    return cls(**kwargs)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate(cfg: Config) -> None:
    """Regras de faixa e coerência entre campos."""
    m, c, i, t, a, o, w, lg = (cfg.model, cfg.camera, cfg.inference, cfg.thermal,
                              cfg.alerts, cfg.outbox, cfg.ws_server, cfg.logging)
    _require(0.0 < m.conf_threshold < 1.0, "model.conf_threshold deve estar entre 0 e 1")
    _require(0.0 < m.iou_threshold < 1.0, "model.iou_threshold deve estar entre 0 e 1")
    _require(1 <= m.max_detections <= 1000, "model.max_detections deve estar entre 1 e 1000")
    _require(1 <= m.intra_op_num_threads <= 4, "model.intra_op_num_threads deve estar entre 1 e 4")
    _require(1 <= m.inter_op_num_threads <= 4, "model.inter_op_num_threads deve estar entre 1 e 4")
    _require(64 <= c.width <= 2592 and 64 <= c.height <= 1944,
             "camera.width/height fora da faixa do sensor (máx. 2592x1944)")
    _require(0.5 <= c.max_fps <= 60, "camera.max_fps deve estar entre 0.5 e 60")
    _require(10 <= c.jpeg_quality <= 100, "camera.jpeg_quality deve estar entre 10 e 100")
    _require(0.5 <= i.capture_interval_s <= 3600, "inference.capture_interval_s deve estar entre 0.5 e 3600 s")
    _require(i.mode in PROCESSING_MODES, f"inference.mode deve ser um de: {', '.join(PROCESSING_MODES)}")
    _require(i.server_timeout_s > 0, "inference.server_timeout_s deve ser > 0")
    _require(m.imgsz == 0 or (32 <= m.imgsz <= 2048 and m.imgsz % 32 == 0),
             "model.imgsz deve ser 0 (automático) ou múltiplo de 32 entre 32 e 2048")
    _require(t.poll_interval_s >= 1, "thermal.poll_interval_s deve ser >= 1")
    _require(30 <= t.temp_high_c <= 85, "thermal.temp_high_c deve estar entre 30 e 85 °C")
    _require(t.temp_resume_c < t.temp_high_c,
             "thermal.temp_resume_c deve ser menor que thermal.temp_high_c (histerese)")
    _require(t.slowdown_factor > 1.0, "thermal.slowdown_factor deve ser > 1")
    _require(t.max_interval_s >= 1.0, "thermal.max_interval_s deve ser >= 1")
    _require(0.0 < a.threshold <= 1.0, "alerts.threshold deve estar entre 0 e 1")
    _require(a.threshold >= m.conf_threshold,
             "alerts.threshold não pode ser menor que model.conf_threshold")
    _require(a.cooldown_s >= 0, "alerts.cooldown_s deve ser >= 0")
    _require(o.max_items >= 1, "outbox.max_items deve ser >= 1")
    _require(o.max_bytes >= 1024 * 1024, "outbox.max_bytes deve ser >= 1 MiB")
    _require(o.max_attempts >= 1, "outbox.max_attempts deve ser >= 1")
    _require(0 < o.backoff_initial_s <= o.backoff_max_s,
             "outbox.backoff_initial_s deve ser > 0 e <= backoff_max_s")
    _require(o.response_timeout_s > 0, "outbox.response_timeout_s deve ser > 0")
    _require(1 <= w.port <= 65535, "ws_server.port inválida")
    _require(w.path.startswith("/"), "ws_server.path deve começar com '/'")
    _require(64 * 1024 <= w.max_frame_bytes <= 32 * 1024 * 1024,
             "ws_server.max_frame_bytes deve estar entre 64 KiB e 32 MiB")
    _require(16 <= w.max_id_bytes <= 4096, "ws_server.max_id_bytes deve estar entre 16 e 4096")
    _require(1 <= w.max_pending_photos <= 64, "ws_server.max_pending_photos deve estar entre 1 e 64")
    _require(lg.level.upper() in {"DEBUG", "INFO", "WARNING", "ERROR"},
             "logging.level deve ser DEBUG, INFO, WARNING ou ERROR")
    _require(lg.max_bytes >= 64 * 1024, "logging.max_bytes deve ser >= 64 KiB")
    _require(0 <= lg.backup_count <= 20, "logging.backup_count deve estar entre 0 e 20")
    _require(lg.metrics_interval_s >= 5, "logging.metrics_interval_s deve ser >= 5")
    if cfg.server_ws_url:
        from .discovery import InvalidServerUrl, normalize_server_url

        try:
            normalize_server_url(cfg.server_ws_url)
        except InvalidServerUrl as exc:
            raise ConfigError(f"server_ws_url: {exc}") from exc


# --------------------------------------------------------------------------- #
#  Carga e gravação
# --------------------------------------------------------------------------- #
def default_path() -> Path:
    env = os.environ.get(CONFIG_ENV_VAR)
    return Path(env) if env else DEFAULT_CONFIG_PATH


def _migrate(data: dict[str, Any]) -> bool:
    """Converte chaves de versões anteriores do config.json. Retorna True se mudou algo."""
    inf = data.get("inference")
    if not isinstance(inf, dict):
        return False
    changed = False
    if "target_fps" in inf:
        fps = inf.pop("target_fps")
        if "capture_interval_s" not in inf and isinstance(fps, (int, float)) and fps > 0:
            inf["capture_interval_s"] = round(1.0 / fps, 3)
        changed = True
    if "offload_when_degraded" in inf:
        offload = inf.pop("offload_when_degraded")
        inf.setdefault("mode", "auto" if offload else "local")
        changed = True
    if "offload_timeout_s" in inf:
        inf.setdefault("server_timeout_s", inf.pop("offload_timeout_s"))
        changed = True
    return changed


def from_dict(data: dict[str, Any], base_dir: Path = Path(".")) -> Config:
    data = json.loads(json.dumps(data))  # cópia profunda
    _migrate(data)
    cfg = _build(Config, data, "")
    cfg.base_dir = base_dir
    validate(cfg)
    return cfg


def load(path: Path | None = None, create: bool = True) -> Config:
    """Lê o config.json. Se não existir e create=True, cria com os padrões.

    Sempre garante um node_id persistente.
    """
    path = Path(path or default_path()).expanduser().resolve()
    if not path.exists():
        if not create:
            raise ConfigError(f"arquivo de configuração não encontrado: {path}")
        cfg = Config(base_dir=path.parent)
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: JSON inválido (linha {exc.lineno}, coluna {exc.colno})") from exc
        except OSError as exc:
            raise ConfigError(f"{path}: não foi possível ler ({exc})") from exc
        cfg = from_dict(data, base_dir=path.parent)
        if _migrate(data):  # grava já no formato novo
            save(cfg, path)
    if not cfg.node_id or not path.exists():
        cfg.node_id = cfg.node_id or str(uuid.uuid4())
        save(cfg, path)
    validate(cfg)
    return cfg


def save(cfg: Config, path: Path | None = None) -> Path:
    """Grava de forma atômica e com permissão 600 (pode conter dados sensíveis)."""
    path = Path(path or (cfg.base_dir / DEFAULT_CONFIG_PATH.name)).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix=".config-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def is_permission_too_open(path: Path) -> bool:
    """True se o arquivo for legível por grupo/outros (somente POSIX)."""
    if os.name != "posix" or not path.exists():
        return False
    return bool(path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO))
