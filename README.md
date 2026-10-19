# pest-node: nó sensor de borda (Raspberry Pi 3)

Software do nó sensor do TCC **"Visão Computacional na Agricultura de Precisão: detectando pragas em dispositivos simples"**.
O nó captura imagens com o módulo de câmera (CSI), detecta pragas com **YOLO26n em ONNX Runtime (só CPU)**,
atende o **app Flutter** pelo mesmo WebSocket do servidor central e envia **alertas** ao **servidor central**.

Hardware de referência (dados reais do Pi do projeto): Raspberry Pi 3 Model B Rev 1.2, Debian 13 (trixie) 64 bits,
Python 3.13, 905 MiB de RAM + 904 MiB de swap, câmera **ov5647** (Camera Module v1.3, 5 MP, foco fixo).

---

## 1. Como o nó conversa com o resto do sistema

Tudo foi derivado do código em `servidor_cv/` e `app_flutter_cv/`, **sem alterá-los**.

| Quem | Fala com | Como |
|---|---|---|
| App Flutter | **nó** (`ws://<ip-do-pi>:8000/ws`) | mesmo protocolo do servidor: o app não muda nada |
| Nó | servidor central (`ws://<ip-servidor>:8000/ws`) | alertas (imagem com praga) e descarga de quadros |

**Protocolo `/ws`** (igual nos dois lados):

- Handshake (texto): `{"type":"ping"}` → `{"type":"pong","connection_id":"<uuid4>"}`
- Quadro binário: `[uint32 Big-Endian = tamanho do id][id em UTF-8][bytes do JPEG]`
- Respostas JSON: `{"id","status":"analisando"|"processado"|"erro","mensagem","analysis","detections"}`,
  com `detections = [{"class_name","confidence","bbox":[x1,y1,x2,y2]}]` em **pixels da imagem original**
  (confiança com 4 casas, bbox com 1 casa, igual ao servidor).
- Ids que começam com `STREAM_`: sem "analisando", e o servidor **não grava** a imagem.

**Alerta = fluxo de persistência do servidor.** O servidor central não tem endpoint para "resultados" ou "alertas".
Quando ele recebe uma foto com id normal (UUID), grava a imagem em `data/<data>/classified|unclassified/` e no
PostgreSQL. Por isso, quando uma detecção passa de `alerts.threshold`, o nó **envia o JPEG do quadro** com id UUID v4.
Quadros sem praga **nunca saem do Pi**. Essa é uma adaptação consciente do requisito "enviar só resultados", decidida
com o autor, já que o contrato real só aceita imagens.

**Descarga de carga (offload).** Quando o Pi está aquecido ou estrangulado e o servidor está acessível, o nó envia o
quadro como `STREAM_xxxxxxxx`: o servidor infere sem gravar, e o nó usa as detecções que voltam.

**Classes:** são 12, lidas dos metadados do próprio ONNX (não ficam fixas no código): Lagarta-da-soja, Joaninha,
Gafanhoto, Lagartas do grupo Heliothinae, Eriopsis connexa, Percevejo, Percevejo-Marrom, Percevejo-verde-ninfa,
Caracol, Besouro Tenebrionidae, Percevejo-verde, Percevejo-verde-ninfa-adulta.

**Modelo:** `models/teste2.onnx` é cópia de `servidor_cv/model/teste2.onnx` (FP32, opset 20, Ultralytics 8.4.120).
Entrada `images [1,3,640,640]`, saída `output0 [1,16,8400]`, `end2end=False`. O pós-processamento (NMS) é escolhido
**pela forma real da saída**: `[1,4+nc,A]`, `[1,A,4+nc]` ou `[1,N,6]` (end-to-end).

---

## 2. Instalação no Raspberry Pi

### 2.1 Copiar o projeto
Copie a pasta `client_raspberry_cv/` para o Pi, por exemplo para `/home/dbrito/Desktop/client_raspberry_cv`.
Use um **usuário comum** (ex.: `dbrito`), não `root`.

### 2.2 Pacotes do sistema (apt)
```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 python3-numpy rpicam-apps libopenblas0 libopenjp2-7 libgomp1
```

### 2.3 Ambiente virtual e bibliotecas Python
```bash
cd ~/Desktop/client_raspberry_cv
bash scripts/install.sh
.venv/bin/pip install -r requirements.txt
```
O `install.sh` confere a arquitetura (avisa se for 32 bits, onde não há wheel do onnxruntime), a versão do Python e os
pacotes apt. Ele cria `.venv` com **`--system-site-packages`**. Isso é necessário porque o **Picamera2 e o libcamera só
existem como pacote apt**, compilados para o Python do sistema; um venv isolado não os enxergaria. Por isso o
`requirements.txt` fixa `numpy==2.2.4`, **a mesma versão do `python3-numpy` do trixie**. Se o pip trocasse o numpy, o
Picamera2 poderia quebrar. O script avisa se as versões divergirem.

`requirements.txt`: `numpy==2.2.4`, `onnxruntime==1.31.0` (tem wheel cp313/aarch64 e suporta o opset 20),
`opencv-python-headless==4.12.0.88`, `websockets==15.0.1`. PyTorch e Ultralytics **não** são instalados no Pi.

### 2.4 Configurar o servidor (QR Code)
Veja a seção 4. Depois:
```bash
scripts/pest-node selftest
```

### 2.5 Serviço systemd (sobe no boot, reinicia em falha)
```bash
bash scripts/install.sh --service
sudo systemctl status pest-node
journalctl -u pest-node -f
```
A unidade (`scripts/pest-node.service`) roda como o seu usuário, nos grupos `video` e `render`, com `Restart=always`,
`RestartSec=5`, sem limite de reinícios e com limites de memória `MemoryHigh=550M` / `MemoryMax=700M`. Ela não espera
a rede no boot: o nó funciona offline e reconecta sozinho.

---

## 3. Uso (CLI)

| Comando | O que faz |
|---|---|
| `scripts/pest-node run` | inicia o nó (é o que o systemd executa) |
| `scripts/pest-node scan-qr [--timeout 120]` | **força** nova leitura do QR e grava `server_ws_url` |
| `scripts/pest-node set-server ws://IP:8000/ws` | grava a URL manualmente |
| `scripts/pest-node selftest` | relatório: plataforma, config, modelo, câmera, sensores, rede, servidor, porta, disco/RAM |
| `scripts/pest-node benchmark [--duration 600]` | benchmark (seção 5) |
| `scripts/pest-node show-config` | mostra a configuração efetiva |

O nó **só pede o QR se não houver URL válida** no `config.json`. Enquanto não houver, cada quadro da câmera também é
examinado em busca de um QR: basta mostrar o QR à câmera com o serviço rodando. Achou, grava e conecta.

**App Flutter:** na tela "Conectar ao Servidor", use `ws://<ip-do-pi>:8000/ws` (ex.: `ws://192.168.5.19:8000/ws`).
O app faz o ping/pong e passa a receber as detecções do Pi. Fotos do app com praga acima do limiar também viram
alerta para o servidor central (`ws_server.forward_alerts`).

---

## 4. QR Code do servidor

Conteúdo: texto puro `ws://<ip-do-servidor>:<porta>/ws`, exatamente o que o `servidor_cv` imprime no terminal ao
iniciar. Também são aceitos `http://` (vira `ws://`) e `https://` (vira `wss://`), e o `/ws` é acrescentado se faltar.
Isso é a mesma regra do app.

Para imprimir um QR em papel, no PC ou no Pi:
```bash
.venv/bin/python tools/make_qr.py ws://192.168.5.12:8000/ws --out qr_servidor.png
```
Dicas para a ov5647 (foco fixo de ~1 m ao infinito): QR **impresso com 10 cm ou mais**, a cerca de **1 m** da lente,
bem iluminado e sem reflexo. QR na tela do terminal também funciona, mas com menos confiabilidade.

**Atenção (servidor):** o QR do servidor usa `SERVER_PORT` do `config.py` (8000), não a porta passada ao `uvicorn`.
Rode o servidor com `--port 8000` (ou ajuste `SERVER_PORT`), senão o QR aponta para a porta errada.

---

## 5. Benchmark (preenche FPS e latência no Pi)

```bash
sudo systemctl stop pest-node      # libera a câmera e a CPU
.venv/bin/python tools/benchmark.py --duration 600                 # 10 min, carga contínua
.venv/bin/python tools/benchmark.py --duration 600 --threads 4     # compara número de threads
.venv/bin/python tools/benchmark.py --duration 600 --adaptive      # com o controle térmico ligado
```
Gera `bench/bench_<data>.csv`, com uma linha por quadro: `capture_ms`, `preprocess_ms`, `infer_ms`, `postprocess_ms`,
`e2e_ms` (captura → detecções prontas), `fps_window`, `cpu_pct`, `rss_mb`, `mem_available_mb`, `temp_c`,
`throttled_hex` e os bits `under_voltage_now`, `freq_capped_now`, `throttled_now`, `soft_temp_limit_now`, além de
`arm_mhz`. Gera também `bench_<data>.summary.json`, com FPS sustentado, latência média/p50/p95/máx, e2e médio/p95,
RSS máximo, temperatura máxima e % de quadros com subtensão ou throttling. **O script só mede**: os números saem do
seu Pi. Sem `--adaptive`, a carga é máxima e contínua, para expor o aquecimento e a subtensão.

Sugestão para o TCC: rode com 1, 2 e 4 threads (`--threads`), depois com e sem dissipador e com fonte boa e fonte
fraca. Compare `throttled_hex`: o bit de subtensão separa "fonte fraca" de "calor".

---

## 6. Configuração (`config.json`)

Criado a partir de `config.example.json` e gravado com permissão `600`. Caminhos relativos partem da pasta do
`config.json`. Chaves desconhecidas ou valores fora da faixa **impedem a inicialização com mensagem clara**
(ex.: `ERRO de configuração: thermal.temp_resume_c deve ser menor que thermal.temp_high_c`).

| Chave | Padrão | Descrição |
|---|---|---|
| `server_ws_url` | `""` | URL do servidor (gravada pelo `scan-qr`) |
| `node_id` | gerado | UUID do nó, criado no 1º uso (aparece nos logs) |
| `model.path` | `models/teste2.onnx` | modelo ONNX |
| `model.conf_threshold` | `0.25` | confiança mínima (igual ao servidor) |
| `model.iou_threshold` | `0.7` | IoU do NMS (padrão do Ultralytics, usado pelo servidor; o app usa 0.45) |
| `model.max_detections` | `300` | máximo de caixas por imagem |
| `model.intra_op_num_threads` | `2` | threads do ONNX Runtime (1–4); 2 deixa CPU livre para câmera e rede |
| `model.inter_op_num_threads` | `1` | paralelismo entre operadores |
| `camera.width` / `height` | `1296` / `972` | modo binado da ov5647 com campo de visão completo |
| `camera.max_fps` | `10` | limite do sensor/ISP (a inferência é bem mais lenta) |
| `camera.jpeg_quality` | `85` | qualidade do JPEG dos alertas/descarga |
| `camera.fake_images_dir` | `""` | se preenchido, usa uma pasta de imagens no lugar da câmera |
| `inference.target_fps` | `0.5` | FPS alvo em condição normal (ajuste pelo benchmark) |
| `inference.offload_when_degraded` | `true` | envia quadros `STREAM_` ao servidor quando aquecido/estrangulado |
| `inference.offload_timeout_s` | `15` | espera máxima pela resposta da descarga |
| `thermal.poll_interval_s` | `5` | intervalo de leitura do `vcgencmd` |
| `thermal.temp_high_c` / `temp_resume_c` | `70` / `65` | limiares com histerese |
| `thermal.react_to_throttling` | `true` | reduz carga com bits de throttling/limite térmico ativos |
| `thermal.react_to_under_voltage` | `true` | reduz carga com subtensão ativa (menos corrente = menos queda) |
| `thermal.slowdown_factor` | `2` | multiplica o intervalo a cada leitura "quente" |
| `thermal.max_interval_s` | `30` | intervalo máximo entre inferências |
| `alerts.threshold` | `0.6` | confiança que dispara alerta |
| `alerts.cooldown_s` | `60` | intervalo mínimo entre alertas da **mesma classe** |
| `alerts.classes` | `[]` | se não vazio, só estas classes geram alerta |
| `outbox.db_path` | `data/outbox.sqlite` | fila persistente de alertas |
| `outbox.max_items` / `max_bytes` | `200` / 100 MiB | limites da fila (descarta os mais antigos, com aviso) |
| `outbox.max_attempts` | `5` | tentativas com resposta "erro" antes de descartar |
| `outbox.backoff_initial_s` / `backoff_max_s` | `2` / `300` | backoff exponencial (com jitter) |
| `outbox.response_timeout_s` | `60` | espera pela resposta "processado" do servidor |
| `ws_server.enabled` / `host` / `port` / `path` | `true` / `0.0.0.0` / `8000` / `/ws` | WebSocket para o app |
| `ws_server.max_frame_bytes` | 5 MiB | maior quadro aceito; acima disso, a conexão é fechada (1009) |
| `ws_server.max_id_bytes` | `256` | maior id aceito |
| `ws_server.max_pending_photos` | `4` | fotos na fila por conexão (acima disso: "erro") |
| `ws_server.forward_alerts` | `true` | fotos do app com praga também viram alerta |
| `logging.level` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `logging.dir` / `max_bytes` / `backup_count` | `logs` / 2 MiB / `3` | log JSON com rotação (máx. ~8 MiB) |
| `logging.metrics_interval_s` | `30` | intervalo do log de métricas |

Não há credenciais: o servidor atual não tem autenticação. Se um dia tiver, o token vai no `config.json` (já `600`)
ou em variável de ambiente, nunca no código.

---

## 7. Projeto do software (escolhas impostas pelo hardware)

- **Memória:** captura sob demanda com `queue=False` e 2 buffers no libcamera (a câmera nunca bloqueia e descarta
  quadros antigos sozinha). Um único buffer numpy de quadro e um tensor de entrada `[1,3,640,640]` são reutilizados.
  Nenhuma lista cresce sem limite: métricas em janelas fixas, fila SQLite limitada e, no streaming do app, só o quadro
  mais recente fica pendente.
- **CPU:** uma única thread de inferência compartilhada entre câmera e app, com prioridade para o app.
- **Pré-processamento idêntico ao Ultralytics:** letterbox centralizado com cinza 114, BGR→RGB, CHW, `/255`. O
  pós-processamento usa NMS por classe e reescala para pixels originais. Validado contra os JSONs gravados pelo
  próprio servidor (seção 9).
- **Térmico:** `vcgencmd measure_temp`, `get_throttled` (todos os bits decodificados) e `measure_clock arm`. Com
  pressão, o intervalo entre inferências dobra (até `max_interval_s`) e, se possível, o quadro é descarregado para o
  servidor. Com alívio (≤ `temp_resume_c` e nenhum bit ativo), o intervalo volta gradualmente.
- **Offline primeiro:** alertas vão para a fila SQLite e são reenviados com backoff exponencial ao reconectar. Um
  item só sai da fila após o "processado" do servidor (entrega "pelo menos uma vez").
- **Logs:** console legível (capturado pelo journald) e arquivo JSON por linha com rotação. Os eventos `metrics`
  (a cada 30 s) trazem FPS, latência, temperatura, bits de throttling, CPU, RSS, RAM livre, fila e conexões.

```
src/pest_node/
  __main__.py   CLI                       runtime.py   orquestração do nó
  config.py     config.json               detector.py  ONNX Runtime + letterbox + NMS
  discovery.py  QR Code e URL             thermal.py   vcgencmd + controle adaptativo
  camera.py     Picamera2 + câmeras falsas client.py   cliente do servidor (alertas, descarga)
  outbox.py     fila SQLite               ws_server.py WebSocket do app
  protocol.py   quadro binário            alerts.py    limiar + cooldown
  models.py     contrato (JSON)           selftest.py, bench.py, sysmetrics.py, logging_setup.py
tools/  export_model.py (PC)  benchmark.py  mock_server.py  make_qr.py
```

---

## 8. Problemas comuns

**Câmera não detectada.** Rode `rpicam-hello --list-cameras`; ela deve listar `ov5647`. Se não listar, desligue o Pi,
reencaixe o cabo flat (contatos virados para o conector HDMI no Pi 3) e confira `camera_auto_detect=1` em
`/boot/firmware/config.txt`. Erro "Picamera2 indisponível": instale `python3-picamera2` e recrie o venv com
`--system-site-packages` (`rm -rf .venv && bash scripts/install.sh`). Erro "Device or resource busy": outro processo
está usando a câmera (o próprio serviço, por exemplo); rode `sudo systemctl stop pest-node`.

**Throttling / FPS caindo.** Rode `vcgencmd get_throttled`. **No Pi do projeto, em repouso, deu `0x50005` (subtensão e
throttling ativos a 53,7 °C, CPU em 600 MHz).** Ou seja, a causa é a **fonte**, não o calor. Use fonte de **5,1 V /
2,5 A** e cabo micro-USB curto e grosso; o ideal é `0x0`. Se for calor (bit 3 / temperatura ≥ 80 °C): dissipador,
ventilação, case aberto. Os logs `load_adjusted` mostram quando e por que o nó reduziu a carga.

**Falta de memória.** Veja `rss_mb` e `mem_available_mb` nos logs `metrics`. Reduza `model.intra_op_num_threads`, desative
o desktop (`sudo systemctl set-default multi-user.target`) e mantenha o swap. O systemd recicla o serviço acima de
700 MB.

**QR não lido.** QR grande (≥ 10 cm), a ~1 m (foco fixo), boa luz, sem reflexo. Teste com
`scripts/pest-node scan-qr --timeout 60`. Alternativa sem QR: `scripts/pest-node set-server ws://IP:8000/ws`.
QR com conteúdo inválido (não-URL, esquema `ftp://` etc.) é ignorado com aviso `qr_invalid` no log.

**Servidor inacessível.** O `selftest` testa TCP, `GET /` e o ping/pong do WebSocket. Confira se o servidor foi iniciado
com `--host 0.0.0.0 --port 8000`, se o firewall do PC libera a porta 8000 e se o Pi está na mesma rede. Enquanto
isso, os alertas ficam na fila (`outbox_items` no log) e são enviados quando o servidor voltar.

**Porta 8000 em uso no Pi.** Outro processo (ou o serviço) já ocupa a porta: `sudo ss -ltnp | grep 8000`.

**`pip install` quer trocar o numpy.** Ajuste `numpy==` no `requirements.txt` para a versão de
`python3 -c "import numpy; print(numpy.__version__)"`.

---

## 9. O que foi validado e onde

**Validado no PC (Windows, Python 3.13, mesmas versões fixadas), sem o hardware do Pi:**
- 144 testes automatizados passando (`pytest`; 140 locais + 4 contra o servidor real; 1 teste de permissão `600` só roda em Linux): protocolo binário (Big-Endian, UTF-8, JPEG, quadros corrompidos, little-endian
  por engano), letterbox/reescala/NMS, os três formatos de saída, config inválida, QR válido/inválido, controle
  térmico com sensor simulado (incluindo a leitura real `0x50005`), fila offline, backoff, retry e queda no meio da
  entrega, WebSocket do app (cabeçalho inválido, JPEG corrompido, quadro grande demais, desconexão no meio da
  mensagem, flood de streaming, caminho errado) e o nó completo com câmera falsa.
- **Fidelidade do detector:** nas 18 imagens que o servidor já tinha processado (`servidor_cv/data/`), o detector do
  nó deu as mesmas classes e a mesma quantidade de caixas, com confiança ±0,01 e caixas ±2 px.
- **Contra o servidor_cv real, sem alterações** (`tests/test_real_server.py`, servidor rodando localmente): ping/pong,
  alerta aceito e **imagem gravada em `classified/`**, descarga `STREAM_`, e resposta do nó idêntica à do servidor
  para o mesmo quadro.
- Formato JSON conferido contra o `model_dump_json()` do Pydantic 2.7.1 do servidor.

**Ainda NÃO validado (precisa do Pi real):** captura com Picamera2/ov5647, leitura de QR pela câmera física, `vcgencmd`
real em carga, FPS/latência/RAM/temperatura (rodar o benchmark), serviço systemd no boot, `install.sh` no trixie,
`selftest` com câmera + modelo + servidor, 10 min contínuos sem estourar RAM, e o app Flutter conectado ao Pi. Também
**não foi validada a gravação no PostgreSQL** (o banco não estava rodando no teste local; o servidor responde
"processado" mesmo quando o banco falha).

Rodar os testes no PC:
```bash
pip install -r requirements-dev.txt
pytest
REAL_SERVER_URL=ws://127.0.0.1:8000/ws pytest tests/test_real_server.py   # com o servidor rodando
```

---

## 10. Observações sobre o contrato (para o autor decidir; nada foi alterado)

1. O servidor não tem endpoint de resultados/alertas: o nó usa a persistência de imagens como alerta (seção 1).
2. O QR do servidor usa `SERVER_PORT`, não a porta real do `uvicorn` (seção 4).
3. O servidor responde "processado" mesmo se a gravação no PostgreSQL falhar. O nó só sabe que a imagem foi
   processada (e gravada em disco), não que entrou no banco.
4. O servidor não limita o tamanho do quadro recebido (o nó limita a 5 MiB).
5. Reenvio após queda pode gerar duplicata no banco (mesmo `packet_id`, mesmo arquivo em disco sobrescrito).
6. O servidor não recebe o id do nó; a rastreabilidade fica no log do nó (`alert_queued`/`alert_delivered` com `packet_id`).
7. O app usa outro ONNX (`assets/models/teste2.onnx`, opset 14, `quantize=16`, Ultralytics 8.4.138) e NMS IoU 0.45.
   O nó e o servidor usam o FP32 com IoU 0.7. Resultados podem diferir levemente entre o app em modo local e o nó.
8. O app não persiste a URL do servidor (fica só em memória) e mostra "10 segundos" no erro de timeout, mas espera 100 s.
9. Nenhum `.pt` foi encontrado na pasta: o `tools/export_model.py` está pronto, mas não foi testado.
