from __future__ import annotations

import numpy as np

from canela.config import Resolution
from canela.detector import _detect_motion, _parse_source


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
        (640, 360): np.full((2, 2), 255, dtype=np.uint8),
        (320, 180): np.zeros((2, 2), dtype=np.uint8),
    }

    resolution, score = _detect_motion(
        frame,
        [Resolution(640, 360), Resolution(320, 180)],
        previous,
        delta_threshold=20,
        motion_ratio_threshold=0.5,
        cv2=_FakeCv2,
    )

    assert resolution == Resolution(320, 180)
    assert score == 1.0
