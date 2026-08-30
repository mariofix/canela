from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ._cv2 import import_cv2
from .alerts import AlertPipeline
from .config import AppConfig, Resolution, StreamConfig
from .evidence import EvidenceWriter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FrameSample:
    timestamp: datetime
    frame: np.ndarray


@dataclass(slots=True)
class DetectionFeed:
    resolution: Resolution
    source: str
    capture: Any


class MotionDetectorService:
    def __init__(self, config: AppConfig):
        self._config = config

    async def run(self) -> None:
        if not self._config.streams:
            raise ValueError("No streams configured")
        pipeline = AlertPipeline(self._config.alerts)
        tasks = [
            asyncio.create_task(self._run_stream(stream, pipeline), name=f"stream:{stream.name}")
            for stream in self._config.streams
        ]
        await asyncio.gather(*tasks)

    async def _run_stream(self, stream: StreamConfig, pipeline: AlertPipeline) -> None:
        cv2 = import_cv2()
        feeds = _open_detection_feeds(stream, cv2)
        primary_feed = feeds[0]

        frame_interval = 1.0 / max(stream.fps, 0.1)
        max_pre_frames = max(1, int(stream.fps * self._config.evidence.pre_seconds))
        pre_buffer: deque[FrameSample] = deque(maxlen=max_pre_frames)

        evidence_writer = EvidenceWriter(
            root_dir=self._resolve_root_dir(),
            output_fps=self._config.evidence.output_fps,
        )

        last_detection_at: datetime | None = None
        previous_frames: dict[tuple[str, int, int], np.ndarray] = {}
        processed_frames = 0

        try:
            while True:
                detection_frames: list[tuple[DetectionFeed, np.ndarray]] = []
                primary_frame: np.ndarray | None = None
                for feed in feeds:
                    ok, frame = feed.capture.read()
                    if not ok:
                        continue
                    if feed is primary_feed:
                        primary_frame = frame
                    detection_frames.append((feed, frame))

                if primary_frame is None:
                    await asyncio.sleep(frame_interval)
                    continue

                now = datetime.now(UTC)
                pre_buffer.append(FrameSample(timestamp=now, frame=primary_frame.copy()))
                processed_frames += 1

                triggered_resolution, score = _detect_motion(
                    detection_frames,
                    previous_frames,
                    self._config.motion.delta_threshold,
                    self._config.motion.motion_ratio_threshold,
                    cv2,
                )
                if _is_in_warmup(processed_frames, self._config.motion.warmup_frames):
                    await asyncio.sleep(frame_interval)
                    continue
                if not triggered_resolution:
                    await asyncio.sleep(frame_interval)
                    continue

                if last_detection_at and now - last_detection_at < timedelta(seconds=self._config.motion.cooldown_seconds):
                    await asyncio.sleep(frame_interval)
                    continue

                last_detection_at = now
                event_frames = [sample.frame for sample in pre_buffer]
                post_frames = await self._collect_post_frames(primary_feed.capture, frame_interval, stream.fps)
                event_frames.extend(post_frames)

                event_dir = evidence_writer.write_event(
                    stream_name=stream.name,
                    detected_at=now,
                    metadata={
                        "trigger_source": triggered_resolution.source or stream.source or "",
                        "trigger_resolution": {
                            "width": triggered_resolution.width or primary_frame.shape[1],
                            "height": triggered_resolution.height or primary_frame.shape[0],
                        },
                        "motion_score": score,
                        "pre_seconds": self._config.evidence.pre_seconds,
                        "post_seconds": self._config.evidence.post_seconds,
                    },
                    frames=event_frames,
                )
                payload: dict[str, Any] = {
                    "stream": stream.name,
                    "detected_at": now.isoformat(),
                    "event_dir": str(event_dir),
                    "motion_score": score,
                }
                await pipeline.run(payload)
                await asyncio.sleep(frame_interval)
        finally:
            for feed in feeds:
                feed.capture.release()

    async def _collect_post_frames(self, capture: Any, frame_interval: float, fps: float) -> list[np.ndarray]:
        post_frame_count = max(0, int(fps * self._config.evidence.post_seconds))
        post_frames: list[np.ndarray] = []
        for _ in range(post_frame_count):
            ok, frame = capture.read()
            if ok:
                post_frames.append(frame.copy())
            await asyncio.sleep(frame_interval)
        return post_frames

    def _resolve_root_dir(self) -> Path:
        from .config import resolve_root

        return resolve_root(self._config.evidence.root_dir)


def _detect_motion(
    detection_frames: list[tuple[DetectionFeed, np.ndarray]],
    previous_frames: dict[tuple[str, int, int], np.ndarray],
    delta_threshold: float,
    motion_ratio_threshold: float,
    cv2: Any,
) -> tuple[Resolution | None, float]:
    for feed, frame in detection_frames:
        resolution = feed.resolution
        if resolution.width is not None and resolution.height is not None:
            processed = cv2.resize(frame, (resolution.width, resolution.height))
        else:
            processed = frame
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        key = (feed.source, gray.shape[1], gray.shape[0])

        prev_gray = previous_frames.get(key)
        previous_frames[key] = gray
        if prev_gray is None:
            continue

        diff = cv2.absdiff(prev_gray, gray)
        changed = (diff >= delta_threshold).sum()
        ratio = float(changed / diff.size)
        if ratio >= motion_ratio_threshold:
            return resolution, ratio
    return None, 0.0


def _parse_source(source: str) -> str | int:
    return int(source) if source.isdigit() else source


def _is_in_warmup(processed_frames: int, warmup_frames: int) -> bool:
    return processed_frames <= max(0, warmup_frames)


def _resolve_feed_source(stream: StreamConfig, resolution: Resolution) -> str:
    source = resolution.source or stream.source
    if not source:
        raise ValueError(
            f"Stream '{stream.name}' requires either stream.source or resolutions[*].source"
        )
    return source


def _open_detection_feeds(stream: StreamConfig, cv2: Any) -> list[DetectionFeed]:
    feeds: list[DetectionFeed] = []
    for resolution in stream.resolutions:
        source = _resolve_feed_source(stream, resolution)
        capture = cv2.VideoCapture(_parse_source(source))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open stream: {stream.name} ({source})")
        feeds.append(DetectionFeed(resolution=resolution, source=source, capture=capture))
    return feeds
