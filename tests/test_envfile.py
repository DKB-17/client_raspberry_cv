"""Testes do .env (envfile.py) e da seleção do modelo pelo .env no config."""

import json
from pathlib import Path

import pytest

from pest_node import config as cfgmod
from pest_node.config import ConfigError
from pest_node.envfile import parse_env, read_env_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(cfgmod.ENV_MODEL_BACKEND, raising=False)
    monkeypatch.delenv(cfgmod.ENV_MODEL_NAME, raising=False)


def test_parse_env_formats():
    text = """
# comentário
PEST_MODEL_BACKEND=litert
export PEST_MODEL_NAME = "meu modelo.tflite"
A='x # nao é comentário'
B=valor # comentário no fim
SEM_IGUAL
"""
    assert parse_env(text) == {"PEST_MODEL_BACKEND": "litert", "PEST_MODEL_NAME": "meu modelo.tflite",
                               "A": "x # nao é comentário", "B": "valor"}


def test_missing_env_file_is_empty(tmp_path):
    assert read_env_file(tmp_path / ".env") == {}


def test_env_example_is_valid():
    env = read_env_file(ROOT / ".env.example")
    assert set(env) == {"PEST_MODEL_BACKEND", "PEST_MODEL_NAME"}
    cfg = cfgmod.from_dict({}, ROOT, env=env)  # tag válida e extensão coerente
    assert cfg.resolve(cfg.model.path) == ROOT / "models" / env["PEST_MODEL_NAME"]


def test_env_file_selects_litert_model_without_touching_config_json(tmp_path):
    (tmp_path / ".env").write_text("PEST_MODEL_BACKEND=litert\nPEST_MODEL_NAME=yolo26n_float32.tflite\n",
                                   encoding="utf-8")
    path = tmp_path / "config.json"
    cfg = cfgmod.load(path)
    assert cfg.model.backend == "litert"
    assert cfg.resolve(cfg.model.path) == tmp_path / "models" / "yolo26n_float32.tflite"
    saved = json.loads(path.read_text(encoding="utf-8"))["model"]
    assert saved["backend"] == "onnx" and saved["path"] == "models/teste2.onnx"


def test_process_env_wins_over_env_file(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PEST_MODEL_BACKEND=litert\nPEST_MODEL_NAME=a.tflite\n", encoding="utf-8")
    monkeypatch.setenv(cfgmod.ENV_MODEL_BACKEND, "onnx")
    monkeypatch.setenv(cfgmod.ENV_MODEL_NAME, "outro.onnx")
    cfg = cfgmod.load(tmp_path / "config.json")
    assert (cfg.model.backend, cfg.model.path) == ("onnx", "models/outro.onnx")


@pytest.mark.parametrize("env, msg", [
    ({"PEST_MODEL_BACKEND": "litert", "PEST_MODEL_NAME": "teste2.onnx"}, "exige arquivo .tflite"),
    ({"PEST_MODEL_BACKEND": "onnx", "PEST_MODEL_NAME": "m.tflite"}, "exige arquivo .onnx"),
    ({"PEST_MODEL_BACKEND": "tensorrt"}, "model.backend"),
])
def test_invalid_env_selection(tmp_path, env, msg):
    with pytest.raises(ConfigError, match=msg):
        cfgmod.from_dict({}, tmp_path, env=env)
