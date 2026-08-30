import logging
import os
import subprocess
import wave
from dataclasses import dataclass

import numpy as np

from ..config import StreamConfig
from .base import StreamDetector

AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SAMPLES = 4096  # ~256ms per chunk at 16kHz


@dataclass
class AudioResult:
    rms: float


class AudioDetector(StreamDetector):
    def __init__(self, cfg: StreamConfig):
        super().__init__(cfg, cooldown=cfg.audio_cooldown)
        self.chunk_bytes = AUDIO_CHUNK_SAMPLES * 2  # 16-bit samples

    @property
    def label(self):
        return f"{self.cfg.name}-audio"

    def open(self):
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.cfg.rtsp_url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", "1",
            "-f", "s16le",
            "-loglevel", "error",
            "pipe:1",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self, handle):
        if handle.poll() is not None:
            return None

        raw = handle.stdout.read(self.chunk_bytes)

        if not raw or len(raw) < self.chunk_bytes:
            return None

        return raw

    def close(self, handle):
        handle.terminate()

    def detect(self, raw):
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2)))

        if rms >= self.cfg.audio_rms_threshold:
            return AudioResult(rms=rms)

        return None

    def save_event(self, raw, result: AudioResult, timestamp):
        logging.info("[%s] rms=%.0f", self.label, result.rms)

        wav_path = os.path.join(self.cfg.save_dir, f"{timestamp}_audio.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(AUDIO_SAMPLE_RATE)
            wf.writeframes(raw)