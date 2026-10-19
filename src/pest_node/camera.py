"""Captura de imagens: Picamera2 (CSI) e câmera falsa para testes.

Estratégia de memória:
  - Captura sob demanda: o laço de inferência pede um quadro quando precisa.
    Com `queue=False` e 2 buffers, o libcamera recicla os quadros sozinho e a
    câmera nunca bloqueia nem acumula fila (quadros antigos são descartados).
  - O quadro é copiado para UM buffer numpy pré-alocado, reutilizado a cada
    captura. Quem chama deve terminar de usar o quadro antes da próxima captura.

Formato: Picamera2 "RGB888" entrega pixels na ordem B,G,R em memória, que é
exatamente o BGR do OpenCV e o que o servidor recebe via cv2.imdecode.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .logging_setup import get_logger, log_event

log = get_logger("camera")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
CAMERA_WARMUP_S = 1.5  # tempo para a exposição/ganho automáticos estabilizarem


class CameraError(RuntimeError):
    pass


class Camera(Protocol):
    width: int
    height: int

    def start(self) -> None: ...

    def capture(self) -> np.ndarray: ...

    def stop(self) -> None: ...


class PiCamera:
    """Módulo de câmera no conector CSI via Picamera2/libcamera."""

    def __init__(self, width: int = 1296, height: int = 972, max_fps: float = 10.0) -> None:
        self.width, self.height, self.max_fps = width, height, max_fps
        self._cam = None
        self._mapped_array = None
        self._buf: np.ndarray | None = None

    def start(self) -> None:
        try:
            from picamera2 import MappedArray, Picamera2  # apt: python3-picamera2
        except ImportError as exc:
            raise CameraError("Picamera2 indisponível. Instale python3-picamera2 pelo apt e crie o venv "
                              "com --system-site-packages.") from exc
        try:
            cam = Picamera2()
        except (IndexError, RuntimeError) as exc:
            raise CameraError(f"nenhuma câmera detectada ({exc}). Verifique o cabo flat e "
                              "`rpicam-hello --list-cameras`.") from exc
        frame_us = int(1_000_000 / self.max_fps)
        config = cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},
            buffer_count=2, queue=False, controls={"FrameDurationLimits": (frame_us, frame_us)})
        cam.configure(config)
        actual_w, actual_h = cam.camera_configuration()["main"]["size"]
        self.width, self.height = int(actual_w), int(actual_h)
        cam.start()
        self._cam, self._mapped_array = cam, MappedArray
        self._buf = np.empty((self.height, self.width, 3), np.uint8)
        discarded = self._warm_up(cam)
        log_event(log, "camera_started", kind="picamera2", size=f"{self.width}x{self.height}",
                  max_fps=self.max_fps, sensor=cam.camera_properties.get("Model"), warmup_frames=discarded)

    @staticmethod
    def _warm_up(cam: object) -> int:
        """Descarta os primeiros quadros enquanto a exposição automática converge.

        Logo após start() a ov5647 entrega quadros escuros (brilho médio ~7 no Pi
        do projeto); sem isso o primeiro quadro analisado sai quase preto.
        """
        deadline = time.monotonic() + CAMERA_WARMUP_S
        count = 0
        while time.monotonic() < deadline:
            cam.capture_request().release()  # type: ignore[attr-defined]
            count += 1
        return count

    def capture(self) -> np.ndarray:
        if self._cam is None or self._buf is None:
            raise CameraError("câmera não iniciada")
        request = self._cam.capture_request()
        try:
            with self._mapped_array(request, "main") as mapped:
                np.copyto(self._buf, mapped.array[:, :self.width, :3])
        finally:
            request.release()
        return self._buf

    def stop(self) -> None:
        if self._cam is not None:
            try:
                self._cam.stop()
                self._cam.close()
            finally:
                self._cam = None


class FakeCamera:
    """Lê imagens de uma pasta em ciclo, redimensionadas para o tamanho configurado."""

    def __init__(self, images_dir: Path, width: int = 1296, height: int = 972, delay_s: float = 0.0) -> None:
        self.images_dir = Path(images_dir)
        self.width, self.height, self.delay_s = width, height, delay_s
        self._files: list[Path] = []
        self._index = 0
        self._buf = np.empty((height, width, 3), np.uint8)
        self.current_file: Path | None = None

    def start(self) -> None:
        if not self.images_dir.is_dir():
            raise CameraError(f"pasta de imagens não encontrada: {self.images_dir}")
        self._files = sorted(p for p in self.images_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self._files:
            raise CameraError(f"nenhuma imagem em {self.images_dir}")
        log_event(log, "camera_started", kind="fake", images=len(self._files), size=f"{self.width}x{self.height}")

    def capture(self) -> np.ndarray:
        if not self._files:
            raise CameraError("câmera falsa não iniciada")
        for _ in range(len(self._files)):
            path = self._files[self._index % len(self._files)]
            self._index += 1
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is not None:
                self.current_file = path
                cv2.resize(img, (self.width, self.height), dst=self._buf, interpolation=cv2.INTER_AREA)
                if self.delay_s:
                    time.sleep(self.delay_s)
                return self._buf
            log_event(log, "fake_image_unreadable", level=30, path=str(path))
        raise CameraError("nenhuma imagem legível na pasta")

    def stop(self) -> None:
        self._files = []


class ArrayCamera:
    """Câmera em memória (testes): devolve os arrays fornecidos em ciclo."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.height, self.width = frames[0].shape[:2]
        self._i = 0

    def start(self) -> None:
        pass

    def capture(self) -> np.ndarray:
        frame = self.frames[self._i % len(self.frames)]
        self._i += 1
        return frame

    def stop(self) -> None:
        pass


def make_camera(width: int, height: int, max_fps: float, fake_images_dir: str | Path | None) -> Camera:
    if fake_images_dir:
        return FakeCamera(Path(fake_images_dir), width, height)
    return PiCamera(width, height, max_fps)
