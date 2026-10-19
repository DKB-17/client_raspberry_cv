"""Benchmark de carga contínua no Pi: FPS, latências, CPU, RAM, temperatura, throttling -> CSV.

Este módulo apenas MEDE; não há números embutidos. Definições:
  - capture_ms:   tempo para obter o quadro da câmera (cópia para o buffer);
  - infer_ms:     somente session.run do ONNX Runtime;
  - e2e_ms:       ponta a ponta = captura + pré-processamento + inferência + pós-processamento;
  - fps_sustained: quadros processados / tempo total medido (após o aquecimento).
Por padrão roda SEM o controle adaptativo (carga máxima contínua), para expor o
efeito do aquecimento/subtensão. Use --adaptive para medir com o controle ligado.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from . import config as cfgmod
from .camera import make_camera
from .detector import Detector
from .sysmetrics import CpuMeter, mem_available_mb, process_rss_mb
from .thermal import AdaptiveController, SimulatedSensor, ThermalSample, VcgencmdSensor

FIELDS = ["timestamp", "elapsed_s", "frame", "warmup", "capture_ms", "preprocess_ms", "infer_ms",
          "postprocess_ms", "e2e_ms", "detections", "fps_window", "sleep_s", "cpu_pct", "rss_mb",
          "mem_available_mb", "temp_c", "throttled_hex", "under_voltage_now", "freq_capped_now",
          "throttled_now", "soft_temp_limit_now", "arm_mhz", "adaptive_interval_s"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pest-node benchmark", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None, help="config.json (padrão: ./config.json)")
    p.add_argument("--duration", type=float, default=600, help="segundos de carga medida (padrão 600 = 10 min)")
    p.add_argument("--warmup", type=int, default=5, help="quadros iniciais descartados das estatísticas")
    p.add_argument("--threads", type=int, default=None, help="sobrescreve model.intra_op_num_threads")
    p.add_argument("--adaptive", action="store_true", help="aplica o controle térmico adaptativo")
    p.add_argument("--sensor-interval", type=float, default=2.0, help="intervalo de leitura do vcgencmd (s)")
    p.add_argument("--images", type=Path, default=None, help="usa câmera falsa com imagens desta pasta")
    p.add_argument("--output", type=Path, default=None, help="CSV de saída (padrão: bench/bench_<data>.csv)")
    return p


def _p(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def summarize(rows: list[dict[str, object]], measured_s: float) -> dict[str, object]:
    data = [r for r in rows if not r["warmup"]]
    col = lambda k: [float(r[k]) for r in data if r[k] not in (None, "")]  # noqa: E731
    infer, e2e = col("infer_ms"), col("e2e_ms")
    flags = lambda k: (100.0 * sum(1 for r in data if r[k] is True) / len(data)) if data else None  # noqa: E731
    return {
        "frames_measured": len(data),
        "duration_s": round(measured_s, 1),
        "fps_sustained": round(len(data) / measured_s, 3) if measured_s > 0 else None,
        "infer_ms_mean": _r(statistics.fmean(infer)) if infer else None,
        "infer_ms_p50": _r(_p(infer, 0.5)),
        "infer_ms_p95": _r(_p(infer, 0.95)),
        "infer_ms_max": _r(max(infer)) if infer else None,
        "e2e_ms_mean": _r(statistics.fmean(e2e)) if e2e else None,
        "e2e_ms_p95": _r(_p(e2e, 0.95)),
        "cpu_pct_mean": _r(statistics.fmean(col("cpu_pct"))) if col("cpu_pct") else None,
        "rss_mb_max": _r(max(col("rss_mb"))) if col("rss_mb") else None,
        "mem_available_mb_min": _r(min(col("mem_available_mb"))) if col("mem_available_mb") else None,
        "temp_c_max": _r(max(col("temp_c"))) if col("temp_c") else None,
        "temp_c_first": _r(col("temp_c")[0]) if col("temp_c") else None,
        "pct_frames_throttled_now": _r(flags("throttled_now")),
        "pct_frames_under_voltage_now": _r(flags("under_voltage_now")),
        "pct_frames_soft_temp_limit_now": _r(flags("soft_temp_limit_now")),
    }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 2)


def run_benchmark(args: argparse.Namespace) -> int:
    cfg = cfgmod.load(args.config)
    m, c = cfg.model, cfg.camera
    threads = args.threads or m.intra_op_num_threads
    det = Detector(cfg.resolve(m.path), m.conf_threshold, m.iou_threshold, m.max_detections, threads,
                   m.inter_op_num_threads)
    images = args.images or (cfg.resolve(c.fake_images_dir) if c.fake_images_dir else None)
    cam = make_camera(c.width, c.height, c.max_fps, images)
    vc = VcgencmdSensor()
    sensor = vc if vc.available else SimulatedSensor([(None, None)])
    ctrl = AdaptiveController(cfg.inference.target_fps, cfg.thermal.temp_high_c, cfg.thermal.temp_resume_c,
                              cfg.thermal.slowdown_factor, cfg.thermal.max_interval_s,
                              cfg.thermal.react_to_throttling, cfg.thermal.react_to_under_voltage)
    out = args.output or cfg.resolve(f"bench/bench_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    cpu = CpuMeter()
    rows: list[dict[str, object]] = []
    recent: list[float] = []
    sample: ThermalSample = sensor.read()
    last_sensor = time.monotonic()

    print(f"Benchmark: {args.duration:.0f} s, {threads} thread(s), câmera {'FALSA' if images else 'Picamera2'}, "
          f"adaptativo={'sim' if args.adaptive else 'não'} -> {out}")
    cam.start()
    try:
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            frame_idx = 0
            t_start = None
            while True:
                now = time.monotonic()
                if t_start is not None and now - t_start >= args.duration:
                    break
                if now - last_sensor >= args.sensor_interval:
                    sample, last_sensor = sensor.read(), now
                    if args.adaptive:
                        ctrl.update(sample)
                t0 = time.perf_counter()
                frame = cam.capture()
                t1 = time.perf_counter()
                res = det.detect(frame)
                t2 = time.perf_counter()
                warm = frame_idx < args.warmup
                if not warm and t_start is None:
                    t_start = time.monotonic()
                recent.append(t2)
                recent = recent[-10:]
                fps_w = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else None
                f = sample.flags
                row = {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "elapsed_s": round(time.monotonic() - t_start, 3) if t_start else 0.0,
                    "frame": frame_idx, "warmup": warm,
                    "capture_ms": round((t1 - t0) * 1e3, 2),
                    "preprocess_ms": round(res.preprocess_ms, 2), "infer_ms": round(res.inference_ms, 2),
                    "postprocess_ms": round(res.postprocess_ms, 2), "e2e_ms": round((t2 - t0) * 1e3, 2),
                    "detections": len(res.detections),
                    "fps_window": round(fps_w, 3) if fps_w else "",
                    "sleep_s": 0.0, "cpu_pct": _r(cpu.percent()), "rss_mb": _r(process_rss_mb()),
                    "mem_available_mb": _r(mem_available_mb()), "temp_c": sample.temp_c,
                    "throttled_hex": f"0x{f.raw:x}" if f else "",
                    "under_voltage_now": f.under_voltage_now if f else "",
                    "freq_capped_now": f.freq_capped_now if f else "",
                    "throttled_now": f.throttled_now if f else "",
                    "soft_temp_limit_now": f.soft_temp_limit_now if f else "",
                    "arm_mhz": sample.arm_freq_mhz, "adaptive_interval_s": round(ctrl.interval_s, 2),
                }
                if args.adaptive:
                    pause = max(0.0, ctrl.interval_s - (time.perf_counter() - t0))
                    row["sleep_s"] = round(pause, 3)
                    time.sleep(pause)
                writer.writerow(row)
                rows.append({k: row[k] for k in ("warmup", "infer_ms", "e2e_ms", "cpu_pct", "rss_mb",
                                                  "mem_available_mb", "temp_c", "under_voltage_now",
                                                  "throttled_now", "soft_temp_limit_now")})
                if frame_idx % 20 == 0:
                    fh.flush()
                    print(f"  quadro {frame_idx}: infer {res.inference_ms:.0f} ms | e2e {(t2 - t0) * 1e3:.0f} ms"
                          f" | temp {sample.temp_c} | {f.describe() if f else 'sem vcgencmd'}")
                frame_idx += 1
    except KeyboardInterrupt:
        print("interrompido; gravando resumo parcial")
    finally:
        cam.stop()
    measured = (time.monotonic() - t_start) if t_start else 0.0
    summary = summarize(rows, measured)
    summary.update({"threads": threads, "adaptive": args.adaptive, "imgsz": det.imgsz,
                    "camera": "fake" if images else "picamera2", "csv": str(out)})
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(build_parser().parse_args(argv))
