from __future__ import annotations

import logging

from canela.cli import _describe_stream, log_startup_summary
from canela.config import AppConfig, EvidenceConfig, MotionConfig, Resolution, StreamConfig


def test_describe_stream_includes_feed_labels_and_sources() -> None:
    stream = StreamConfig(
        name="camera-a",
        source=None,
        fps=5,
        resolutions=[
            Resolution(name="high", source="rtsp://127.0.0.1:554/s0"),
            Resolution(name="low", source="rtsp://127.0.0.1:554/s1"),
        ],
    )

    description = _describe_stream(stream)

    assert "camera-a" in description
    assert "high=rtsp://127.0.0.1:554/s0" in description
    assert "low=rtsp://127.0.0.1:554/s1" in description


def test_log_startup_summary_emits_friendly_runtime_messages(caplog) -> None:
    config = AppConfig(
        streams=[
            StreamConfig(
                name="front-door",
                source="rtsp://127.0.0.1:554/s0",
                fps=5,
                resolutions=[Resolution(name="high", source="rtsp://127.0.0.1:554/s0")],
            )
        ],
        motion=MotionConfig(warmup_frames=30),
        evidence=EvidenceConfig(pre_seconds=5, post_seconds=5),
        alerts=[],
    )

    with caplog.at_level(logging.INFO):
        log_startup_summary(config, ["settings.toml"], "INFO")

    text = caplog.text
    assert "Canela is starting" in text
    assert "Configured streams: 1" in text
    assert "Ready. Waiting for motion events" in text
