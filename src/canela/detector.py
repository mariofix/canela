from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
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
    subtractor: Any = field(default=None)


class MotionDetectorService:
    def __init__(self, config: AppConfig):
        self._config = config

    async def run(self) -> None:
        if not self._config.streams:
            raise ValueError("No streams configured")
        logger.info("Starting motion detector service for %d stream(s).", len(self._config.streams))
        pipeline = AlertPipeline(self._config.alerts)
        tasks = [
            asyncio.create_task(self._run_stream(stream, pipeline), name=f"stream:{stream.name}")
            for stream in self._config.streams
        ]
        await asyncio.gather(*tasks)

    async def _run_stream(self, stream: StreamConfig, pipeline: AlertPipeline) -> None:
        cv2 = import_cv2()
        motion_cfg = self._config.motion

        feeds: list[DetectionFeed] | None = None

        try:
            frame_interval = 1.0 / max(stream.fps, 0.1)
            max_pre_frames = max(1, int(stream.fps * self._config.evidence.pre_seconds))
            pre_buffer: deque[FrameSample] = deque(maxlen=max_pre_frames)

            evidence_writer = EvidenceWriter(
                root_dir=self._resolve_root_dir(),
                output_fps=self._config.evidence.output_fps,
            )

            last_detection_at: datetime | None = None
            prev_primary_frame: np.ndarray | None = None
            processed_frames = 0
            warmup_complete_logged = False

            while True:
                # (Re)open feeds if needed
                if feeds is None:
                    logger.info("Stream '%s': (re)opening feeds.", stream.name)
                    feeds = _open_detection_feeds(stream, cv2, motion_cfg)
                    processed_frames = 0
                    warmup_complete_logged = False
                    logger.info(
                        "Stream '%s' online with %d feed(s) at %.2f fps.",
                        stream.name,
                        len(feeds),
                        stream.fps,
                    )

                detection_frames: list[tuple[DetectionFeed, np.ndarray]] = []
                read_failed = False
                for feed in feeds:
                    ok, frame = feed.capture.read()
                    if not ok:
                        logger.debug("No frame from stream '%s' feed '%s'.", stream.name, feed.source)
                        read_failed = True
                        break
                    detection_frames.append((feed, frame))

                if read_failed or not detection_frames:
                    retry_delay = stream.reconnect_backoff_seconds
                    logger.warning(
                        "Primary feed unavailable for stream '%s'; reconnecting in %.1f seconds.",
                        stream.name,
                        retry_delay,
                    )
                    for feed in feeds:
                        feed.capture.release()
                    feeds = None
                    await asyncio.sleep(retry_delay)
                    continue

                primary_feed, primary_frame = _select_primary_feed(detection_frames)
                now = datetime.now(UTC)
                pre_buffer.append(FrameSample(timestamp=now, frame=primary_frame.copy()))
                processed_frames += 1

                triggered_resolution, score, motion_boxes = _detect_motion_mog2(
                    detection_frames,
                    motion_cfg.min_contour_area,
                    cv2,
                    primary_shape=primary_frame.shape[:2],
                )

                if _is_in_warmup(processed_frames, motion_cfg.warmup_frames):
                    logger.debug(
                        "Warm-up stream '%s': frame %d/%d.",
                        stream.name,
                        processed_frames,
                        max(0, motion_cfg.warmup_frames),
                    )
                    prev_primary_frame = primary_frame.copy()
                    await asyncio.sleep(frame_interval)
                    continue

                if not warmup_complete_logged:
                    logger.info(
                        "Stream '%s' warm-up complete after %d frame(s). Motion alerts are now active.",
                        stream.name,
                        max(0, motion_cfg.warmup_frames),
                    )
                    warmup_complete_logged = True

                if not triggered_resolution:
                    prev_primary_frame = primary_frame.copy()
                    await asyncio.sleep(frame_interval)
                    continue

                if last_detection_at and now - last_detection_at < timedelta(seconds=motion_cfg.cooldown_seconds):
                    logger.debug("Stream '%s' is in cooldown; motion trigger suppressed.", stream.name)
                    prev_primary_frame = primary_frame.copy()
                    await asyncio.sleep(frame_interval)
                    continue

                last_detection_at = now
                trigger_source = triggered_resolution.source or stream.source or primary_feed.source
                logger.info(
                    "🚨 Motion detected on stream '%s' (source=%s, score=%.4f).",
                    stream.name,
                    trigger_source,
                    score,
                )
                event_frames = [sample.frame for sample in pre_buffer]
                post_frames = await self._collect_post_frames(primary_feed.capture, frame_interval, stream.fps)
                event_frames.extend(post_frames)

                event_dir = evidence_writer.write_event(
                    stream_name=stream.name,
                    detected_at=now,
                    metadata={
                        "trigger_source": trigger_source,
                        "trigger_resolution": {
                            "width": triggered_resolution.width or primary_frame.shape[1],
                            "height": triggered_resolution.height or primary_frame.shape[0],
                        },
                        "motion_score": score,
                        "pre_seconds": self._config.evidence.pre_seconds,
                        "post_seconds": self._config.evidence.post_seconds,
                    },
                    frames=event_frames,
                    detection_frame=primary_frame,
                    motion_boxes=motion_boxes,
                    prev_frame=prev_primary_frame,
                )
                logger.info("Evidence captured for stream '%s' at: %s", stream.name, event_dir)
                payload: dict[str, Any] = {
                    "stream": stream.name,
                    "detected_at": now.isoformat(),
                    "event_dir": str(event_dir),
                    "motion_score": score,
                }
                await pipeline.run(payload)
                logger.info("Alert pipeline completed for stream '%s'.", stream.name)
                prev_primary_frame = primary_frame.copy()
                await asyncio.sleep(frame_interval)
        finally:
            if feeds is not None:
                for feed in feeds:
                    feed.capture.release()
            logger.info("Stream '%s' stopped.", stream.name)

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


def _detect_motion_mog2(
    detection_frames: list[tuple[DetectionFeed, np.ndarray]],
    min_contour_area: int,
    cv2: Any,
    primary_shape: tuple[int, int] | None = None,
) -> tuple[Resolution | None, float, list[tuple[int, int, int, int]]]:
    """Apply MOG2 background subtraction on each detection feed.

    Returns the triggering resolution, a motion score (ratio of changed pixels),
    and the list of bounding boxes (in primary/high-res frame coordinates).
    """
    for feed, frame in detection_frames:
        resolution = feed.resolution
        if resolution.width is not None and resolution.height is not None:
            detect_frame = cv2.resize(frame, (resolution.width, resolution.height), interpolation=cv2.INTER_AREA)
        else:
            detect_frame = frame

        detect_h, detect_w = detect_frame.shape[:2]
        if primary_shape is None:
            primary_h, primary_w = frame.shape[:2]
        else:
            primary_h, primary_w = primary_shape
        scale_x = primary_w / detect_w
        scale_y = primary_h / detect_h

        mask = feed.subtractor.apply(detect_frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_boxes: list[tuple[int, int, int, int]] = []
        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            total_area += area
            x, y, w, h = cv2.boundingRect(contour)
            hx = int(x * scale_x)
            hy = int(y * scale_y)
            hw = int(w * scale_x)
            hh = int(h * scale_y)
            motion_boxes.append((hx, hy, hw, hh))

        if motion_boxes:
            ratio = float(total_area / (detect_w * detect_h))
            return resolution, ratio, motion_boxes

    return None, 0.0, []


def _select_primary_feed(
    detection_frames: list[tuple[DetectionFeed, np.ndarray]],
) -> tuple[DetectionFeed, np.ndarray]:
    return max(
        detection_frames,
        key=lambda item: (
            item[1].shape[0] * item[1].shape[1],
            item[1].shape[1],
            item[1].shape[0],
        ),
    )


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


def _open_detection_feeds(stream: StreamConfig, cv2: Any, motion_cfg: Any) -> list[DetectionFeed]:
    if not stream.resolutions:
        raise ValueError(f"Stream '{stream.name}' has no configured resolutions/feeds")
    feeds: list[DetectionFeed] = []
    for resolution in stream.resolutions:
        source = _resolve_feed_source(stream, resolution)
        capture = cv2.VideoCapture(_parse_source(source))
        if not capture.isOpened():
            capture.release()
            for feed in feeds:
                feed.capture.release()
            raise RuntimeError(f"Unable to open stream: {stream.name} ({source})")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        subtractor = cv2.createBackgroundSubtractorMOG2(
            history=motion_cfg.mog2_history,
            varThreshold=motion_cfg.mog2_var_threshold,
            detectShadows=False,
        )
        feeds.append(DetectionFeed(resolution=resolution, source=source, capture=capture, subtractor=subtractor))
    return feeds
