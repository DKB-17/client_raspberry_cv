# Nó sensor de pragas (Raspberry Pi)

Comandos para instalar e rodar no Raspberry Pi, de dentro da pasta do projeto.

## 1. Instalar as bibliotecas

```bash
sudo apt update
sudo apt install -y python3-venv python3-picamera2
```

```bash
cd /home/dbrito/Desktop/client_raspberry_cv
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
```

## 2. Rodar

Primeira vez (lê o QR Code do servidor com a câmera):

```bash
.venv/bin/python main.py --scan-qr
```

Próximas vezes:

```bash
.venv/bin/python main.py
```

## Outros comandos

```bash
.venv/bin/python main.py --selftest
.venv/bin/python main.py --set-server "ws://192.168.5.12:8000/ws?capture_interval=5"
.venv/bin/python main.py --show-config
```

## Trocando o modelo (ONNX ou LiteRT)

O modelo é escolhido pelo `.env` (ao lado do `config.json`). Crie a partir do exemplo:

```bash
cp .env.example .env
```

| Tag (`PEST_MODEL_BACKEND`) | Arquivo  | Runtime                                  |
|----------------------------|----------|------------------------------------------|
| `onnx`                     | `.onnx`  | ONNX Runtime                             |
| `litert`                   | `.tflite`| LiteRT (`ai-edge-litert`), export `tflite` do Ultralytics |

`PEST_MODEL_NAME` é o nome do arquivo dentro de `models/`. Para usar outro modelo, copie o arquivo
para `models/`, ajuste as duas linhas do `.env` e reinicie. Exemplo LiteRT:

```bash
PEST_MODEL_BACKEND=litert
PEST_MODEL_NAME=yolo26n_float32.tflite
```

Para gerar o `.tflite` no PC (com Ultralytics instalado):

```bash
python tools/export_model.py best.pt --format litert
```

Confira o modelo carregado com `.venv/bin/python main.py --selftest` (linha "Modelo").
