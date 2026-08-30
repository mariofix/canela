from .audio import AudioDetector
from .base import StreamDetector
from .motion import MotionDetector

__all__ = ["StreamDetector", "MotionDetector", "AudioDetector", "build_detectors"]


def build_detectors(stream_configs):
    detectors = []

    for cfg in stream_configs:
        detectors.append(MotionDetector(cfg))

        if cfg.audio_enabled:
            detectors.append(AudioDetector(cfg))

    return detectors