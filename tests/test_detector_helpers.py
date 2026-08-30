from __future__ import annotations

import numpy as np

from canela.config import Resolution, StreamConfig
from canela.detector import DetectionFeed, _detect_motion, _is_in_warmup, _parse_source, _resolve_feed_source


class _FakeCv2:
    COLOR_BGR2GRAY = 0

    @staticmethod
    def resize(frame: np.ndarray, _size: tuple[int, int]) -> np.ndarray:
        return frame

    @staticmethod
    def cvtColor(frame: np.ndarray, _color: int) -> np.ndarray:
        return frame[..., 0]

    @staticmethod
    def absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)


def test_parse_source_supports_webcam_index() -> None:
    assert _parse_source("0") == 0
    assert _parse_source("rtsp://camera") == "rtsp://camera"


def test_detect_motion_uses_all_configured_resolutions() -> None:
    frame = np.full((2, 2, 3), 255, dtype=np.uint8)
    previous = {
        ("rtsp://a", 2, 2): np.full((2, 2), 255, dtype=np.uint8),
        ("rtsp://b", 2, 2): np.zeros((2, 2), dtype=np.uint8),
    }
    feed_a = DetectionFeed(resolution=Resolution(width=640, height=360), source="rtsp://a", capture=None)
    feed_b = DetectionFeed(resolution=Resolution(width=320, height=180), source="rtsp://b", capture=None)

    resolution, score = _detect_motion(
        [(feed_a, frame), (feed_b, frame)],
        previous,
        delta_threshold=20,
        motion_ratio_threshold=0.5,
        cv2=_FakeCv2,
    )

    assert resolution == Resolution(width=320, height=180)
    assert score == 1.0


def test_warmup_frame_gate() -> None:
    assert _is_in_warmup(1, 30) is True
    assert _is_in_warmup(30, 30) is True
    assert _is_in_warmup(31, 30) is False
    assert _is_in_warmup(1, 0) is False


def test_resolve_feed_source_supports_per_resolution_sources() -> None:
    stream = StreamConfig(name="camera-a", source=None, resolutions=[Resolution(source="rtsp://127.0.0.1:554/s1")])
    assert _resolve_feed_source(stream, stream.resolutions[0]) == "rtsp://127.0.0.1:554/s1"
