import json
import os
import stat

import pytest

from pest_node import config as cfgmod
from pest_node.config import ConfigError


def test_load_creates_defaults_with_node_id(tmp_path):
    path = tmp_path / "config.json"
    cfg = cfgmod.load(path)
    assert path.exists()
    assert cfg.node_id
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["node_id"] == cfg.node_id
    assert data["ws_server"]["port"] == 8000
    # segunda carga reutiliza o mesmo node_id
    assert cfgmod.load(path).node_id == cfg.node_id


def test_relative_paths_resolve_from_config_dir(tmp_path):
    cfg = cfgmod.load(tmp_path / "config.json")
    assert cfg.resolve(cfg.model.path) == tmp_path / "models" / "teste2.onnx"


@pytest.mark.skipif(os.name != "posix", reason="permissões POSIX")
def test_saved_with_mode_600(tmp_path):
    path = tmp_path / "config.json"
    cfgmod.load(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("patch, msg", [
    ({"model": {"conf_threshold": 1.5}}, "model.conf_threshold"),
    ({"thermal": {"temp_high_c": 60, "temp_resume_c": 65}}, "temp_resume_c"),
    ({"ws_server": {"port": "8000"}}, "ws_server.port"),
    ({"camera": {"jpeg_quality": True}}, "camera.jpeg_quality"),
    ({"alerts": {"threshold": 0.1}}, "alerts.threshold"),
    ({"model": {"nao_existe": 1}}, "nao_existe"),
    ({"server_ws_url": "ftp://x"}, "server_ws_url"),
    ({"logging": {"level": "VERBOSE"}}, "logging.level"),
])
def test_invalid_values_raise_clear_error(tmp_path, patch, msg):
    with pytest.raises(ConfigError, match=msg.replace(".", r"\.")):
        cfgmod.from_dict(patch, tmp_path)


def test_malformed_json(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ invalido", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON inválido"):
        cfgmod.load(path)


def test_example_config_is_valid():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "config.example.json"
    data = json.loads(example.read_text(encoding="utf-8"))
    cfgmod.from_dict(data, example.parent)
