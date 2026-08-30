from __future__ import annotations

import numpy as np

from canela.config import Resolution, StreamConfig
from canela.detector import (
    DetectionFeed,
    _detect_motion_mog2,
    _is_in_warmup,
    _parse_source,
    _resolve_feed_source,
    _select_primary_feed,
)


class _FakeSubtractor:
    """Fake MOG2 background subtractor that returns a full white mask (all motion)."""

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return np.full(frame.shape[:2], 255, dtype=np.uint8)


class _FakeCv2:
    INTER_AREA = 0
    THRESH_BINARY = 0
    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 0

    @staticmethod
    def resize(frame: np.ndarray, size: tuple[int, int], interpolation: int = 0) -> np.ndarray:
        # Return a frame of the requested size for testing
        h, w = size[1], size[0]
        return np.zeros((h, w, 3), dtype=np.uint8)

    @staticmethod
    def threshold(mask: np.ndarray, thresh: float, maxval: float, type_: int) -> tuple[float, np.ndarray]:
        return thresh, mask

    @staticmethod
    def GaussianBlur(mask: np.ndarray, ksize: tuple[int, int], sigma: float) -> np.ndarray:
        return mask

    @staticmethod
    def findContours(mask: np.ndarray, mode: int, method: int) -> tuple[list[np.ndarray], None]:
        # Return one contour covering the full mask area
        h, w = mask.shape[:2]
        contour = np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]], dtype=np.int32)
        return [contour], None

    @staticmethod
    def contourArea(contour: np.ndarray) -> float:
        return float(contour.shape[0] * 10)

    @staticmethod
    def boundingRect(contour: np.ndarray) -> tuple[int, int, int, int]:
        xs = contour[:, 0, 0]
        ys = contour[:, 0, 1]
        x, y = int(xs.min()), int(ys.min())
        w = int(xs.max()) - x + 1
        h = int(ys.max()) - y + 1
        return x, y, w, h


def test_parse_source_supports_webcam_index() -> None:
    assert _parse_source("0") == 0
    assert _parse_source("rtsp://camera") == "rtsp://camera"


def test_detect_motion_mog2_returns_motion_when_contours_exceed_min_area() -> None:
    frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    feed = DetectionFeed(
        resolution=Resolution(width=640, height=360),
        source="rtsp://a",
        capture=None,
        subtractor=_FakeSubtractor(),
    )

    resolution, score, boxes = _detect_motion_mog2(
        [(feed, frame)],
        min_contour_area=1,  # accept any contour
        cv2=_FakeCv2,
    )

    assert resolution == Resolution(width=640, height=360)
    assert score > 0
    assert len(boxes) > 0


def test_detect_motion_mog2_scales_boxes_to_primary_shape() -> None:
    frame = np.full((180, 320, 3), 128, dtype=np.uint8)
    feed = DetectionFeed(
        resolution=Resolution(width=320, height=180),
        source="rtsp://a",
        capture=None,
        subtractor=_FakeSubtractor(),
    )

    resolution, score, boxes = _detect_motion_mog2(
        [(feed, frame)],
        min_contour_area=1,
        cv2=_FakeCv2,
        primary_shape=(360, 640),
    )

    assert resolution == Resolution(width=320, height=180)
    assert score > 0
    assert boxes == [(0, 0, 640, 360)]


def test_detect_motion_mog2_skips_small_contours() -> None:
    frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    feed = DetectionFeed(
        resolution=Resolution(width=640, height=360),
        source="rtsp://a",
        capture=None,
        subtractor=_FakeSubtractor(),
    )

    resolution, score, boxes = _detect_motion_mog2(
        [(feed, frame)],
        min_contour_area=10_000_000,  # impossibly large — nothing should pass
        cv2=_FakeCv2,
    )

    assert resolution is None
    assert score == 0.0
    assert boxes == []


def test_warmup_frame_gate() -> None:
    assert _is_in_warmup(1, 30) is True
    assert _is_in_warmup(30, 30) is True
    assert _is_in_warmup(31, 30) is False
    assert _is_in_warmup(1, 0) is False


def test_resolve_feed_source_supports_per_resolution_sources() -> None:
    stream = StreamConfig(name="camera-a", source=None, resolutions=[Resolution(source="rtsp://127.0.0.1:554/s1")])
    assert _resolve_feed_source(stream, stream.resolutions[0]) == "rtsp://127.0.0.1:554/s1"


def test_select_primary_feed_prefers_highest_resolution_frame() -> None:
    high_feed = DetectionFeed(resolution=Resolution(name="high"), source="rtsp://high", capture=None)
    low_feed = DetectionFeed(resolution=Resolution(name="low"), source="rtsp://low", capture=None)
    low_frame = np.zeros((180, 320, 3), dtype=np.uint8)
    high_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    primary_feed, primary_frame = _select_primary_feed([(low_feed, low_frame), (high_feed, high_frame)])

    assert primary_feed is high_feed
    assert primary_frame is high_frame
