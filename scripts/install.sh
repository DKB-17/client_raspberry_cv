#!/usr/bin/env bash
# Instalação do pest-node no Raspberry Pi. Idempotente: pode rodar várias vezes.
#
#   bash scripts/install.sh              # verifica sistema, cria venv e config.json
#   bash scripts/install.sh --apt        # + instala pacotes do sistema (sudo apt)
#   bash scripts/install.sh --pip        # + pip install -r requirements.txt no venv
#   bash scripts/install.sh --service    # + instala e habilita o serviço systemd (sudo)
#   bash scripts/install.sh --all        # tudo acima
#
# Por padrão NÃO instala bibliotecas: apenas mostra os comandos, para você rodar.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
DO_APT=0; DO_PIP=0; DO_SERVICE=0
for arg in "$@"; do
  case "$arg" in
    --apt) DO_APT=1 ;;
    --pip) DO_PIP=1 ;;
    --service) DO_SERVICE=1 ;;
    --all) DO_APT=1; DO_PIP=1; DO_SERVICE=1 ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $arg" >&2; exit 2 ;;
  esac
done

APT_PACKAGES=(python3-venv python3-pip python3-picamera2 python3-numpy rpicam-apps
              libopenblas0 libopenjp2-7 libgomp1)

info() { printf '\033[32m[ok]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[aviso]\033[0m %s\n' "$*"; }
fail() { printf '\033[31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

# ----------------------------------------------------------------- sistema
ARCH="$(uname -m)"
BITS="$(getconf LONG_BIT)"
if [[ "$ARCH" != "aarch64" || "$BITS" != "64" ]]; then
  warn "Sistema $ARCH/${BITS} bits. O onnxruntime oficial só tem wheels para 64 bits (aarch64)."
  warn "Use o Raspberry Pi OS / Debian 64 bits; em 32 bits o pip install vai falhar."
else
  info "Arquitetura $ARCH (64 bits)"
fi
if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  info "Sistema: ${PRETTY_NAME:-desconhecido}"
  [[ "${VERSION_CODENAME:-}" =~ ^(bookworm|trixie)$ ]] || warn "Testado para bookworm/trixie; você tem ${VERSION_CODENAME:-?}"
fi
PY="$(command -v python3 || true)"
[[ -n "$PY" ]] || fail "python3 não encontrado"
PYVER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || fail "Python >= 3.11 necessário (encontrado $PYVER)"
info "Python $PYVER"

# ----------------------------------------------------------------- apt
if (( DO_APT )); then
  sudo apt-get update
  sudo apt-get install -y "${APT_PACKAGES[@]}"
else
  missing=()
  for p in "${APT_PACKAGES[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
  if (( ${#missing[@]} )); then
    warn "Pacotes do sistema ausentes: ${missing[*]}"
    echo "      rode: sudo apt install -y ${missing[*]}"
  else
    info "Pacotes do sistema presentes"
  fi
fi

# ----------------------------------------------------------------- venv
# --system-site-packages: o Picamera2 (e o libcamera) só existem como pacote apt,
# compilados para o Python do sistema; sem essa opção o venv não os enxerga.
if [[ ! -x "$DIR/.venv/bin/python" ]]; then
  "$PY" -m venv --system-site-packages "$DIR/.venv"
  info "venv criado em $DIR/.venv (com --system-site-packages)"
else
  info "venv já existe em $DIR/.venv"
fi
if (( DO_PIP )); then
  "$DIR/.venv/bin/python" -m pip install --upgrade pip
  "$DIR/.venv/bin/python" -m pip install -r "$DIR/requirements.txt"
else
  echo "      bibliotecas Python: $DIR/.venv/bin/pip install -r $DIR/requirements.txt"
fi
SYS_NUMPY="$("$PY" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '?')"
REQ_NUMPY="$(sed -n 's/^numpy==//p' "$DIR/requirements.txt")"
[[ "$SYS_NUMPY" == "$REQ_NUMPY" || "$SYS_NUMPY" == "?" ]] || \
  warn "numpy do apt é $SYS_NUMPY, requirements.txt fixa $REQ_NUMPY: ajuste a linha numpy== para $SYS_NUMPY"

# ----------------------------------------------------------------- config e pastas
mkdir -p "$DIR/data" "$DIR/logs" "$DIR/bench"
if [[ ! -f "$DIR/config.json" ]]; then
  cp "$DIR/config.example.json" "$DIR/config.json"
  info "config.json criado a partir de config.example.json"
fi
chmod 600 "$DIR/config.json"
chmod +x "$DIR/scripts/pest-node"
[[ -f "$DIR/models/teste2.onnx" ]] || warn "modelo ausente: copie o .onnx para $DIR/models/"
if [[ "$(id -u)" -eq 0 && "$RUN_USER" != "root" ]]; then
  chown -R "$RUN_USER":"$RUN_USER" "$DIR/data" "$DIR/logs" "$DIR/bench" "$DIR/config.json" "$DIR/.venv"
fi

# ----------------------------------------------------------------- systemd
if (( DO_SERVICE )); then
  [[ "$RUN_USER" != "root" ]] || warn "o serviço vai rodar como root; prefira um usuário comum"
  sudo usermod -aG video,render "$RUN_USER" || true
  tmp="$(mktemp)"
  sed -e "s|@USER@|$RUN_USER|g" -e "s|@DIR@|$DIR|g" "$DIR/scripts/pest-node.service" > "$tmp"
  sudo install -m 644 "$tmp" /etc/systemd/system/pest-node.service
  rm -f "$tmp"
  sudo systemctl daemon-reload
  sudo systemctl enable pest-node.service
  sudo systemctl restart pest-node.service
  info "serviço instalado: sudo systemctl status pest-node | journalctl -u pest-node -f"
else
  echo "      serviço systemd: bash scripts/install.sh --service"
fi

echo
info "Próximos passos: scripts/pest-node scan-qr  e  scripts/pest-node selftest"
