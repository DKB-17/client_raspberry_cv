"""Leitura mínima de arquivo .env (KEY=VALUE), sem dependências externas.

Aceita comentários (#), linhas vazias, prefixo `export ` e valores entre aspas.
Não altera os.environ: quem chama decide a precedência.
"""

from __future__ import annotations

from pathlib import Path


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()  # comentário no fim da linha
        values[key] = value
    return values


def read_env_file(path: Path) -> dict[str, str]:
    """Conteúdo do .env, ou {} se o arquivo não existir."""
    try:
        return parse_env(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
