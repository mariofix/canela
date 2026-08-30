from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf


@dataclass(slots=True)
class Resolution:
    width: int | None = None
    height: int | None = None
    source: str | None = None
    name: str | None = None


@dataclass(slots=True)
class StreamConfig:
    name: str
    source: str | None = None
    fps: float = 5.0
    reconnect_backoff_seconds: float = 5.0
    resolutions: list[Resolution] = field(default_factory=lambda: [Resolution(width=640, height=360)])


@dataclass(slots=True)
class MotionConfig:
    delta_threshold: float = 20.0
    motion_ratio_threshold: float = 0.02
    cooldown_seconds: float = 10.0
    warmup_frames: int = 30
    mog2_history: int = 300
    mog2_var_threshold: float = 32.0
    min_contour_area: int = 500


@dataclass(slots=True)
class EvidenceConfig:
    root_dir: str = "evidence"
    pre_seconds: float = 5.0
    post_seconds: float = 5.0
    output_fps: float = 10.0


@dataclass(slots=True)
class AlertStep:
    run: str
    async_step: bool = False
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    streams: list[StreamConfig]
    motion: MotionConfig = field(default_factory=MotionConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    alerts: list[AlertStep] = field(default_factory=list)


def _as_resolution(value: dict[str, Any]) -> Resolution:
    width = value.get("width")
    height = value.get("height")
    source = value.get("source")
    name = value.get("name")
    return Resolution(
        width=int(width) if width is not None else None,
        height=int(height) if height is not None else None,
        source=str(source) if source is not None else None,
        name=str(name) if name is not None else None,
    )


def load_config(settings_files: list[str] | None = None) -> AppConfig:
    files = settings_files or ["settings.toml", ".secrets.toml"]
    settings = Dynaconf(settings_files=files, environments=True)

    raw_streams = settings.get("streams") or []
    streams: list[StreamConfig] = []
    for item in raw_streams:
        name = str(item["name"])
        source_value = item.get("source")
        source = str(source_value) if source_value is not None else None
        fps = float(item.get("fps", 5.0))
        reconnect_backoff_seconds = float(item.get("reconnect_backoff_seconds", 5.0))
        raw_resolutions = item.get("resolutions") or [{"width": 640, "height": 360}]
        resolutions = [_as_resolution(entry) for entry in raw_resolutions]
        streams.append(
            StreamConfig(
                name=name,
                source=source,
                fps=fps,
                reconnect_backoff_seconds=reconnect_backoff_seconds,
                resolutions=resolutions,
            )
        )

    motion_data = settings.get("motion") or {}
    motion_defaults = MotionConfig()
    motion = MotionConfig(
        delta_threshold=float(motion_data.get("delta_threshold", motion_defaults.delta_threshold)),
        motion_ratio_threshold=float(
            motion_data.get("motion_ratio_threshold", motion_defaults.motion_ratio_threshold)
        ),
        cooldown_seconds=float(motion_data.get("cooldown_seconds", motion_defaults.cooldown_seconds)),
        warmup_frames=int(motion_data.get("warmup_frames", motion_defaults.warmup_frames)),
        mog2_history=int(motion_data.get("mog2_history", motion_defaults.mog2_history)),
        mog2_var_threshold=float(motion_data.get("mog2_var_threshold", motion_defaults.mog2_var_threshold)),
        min_contour_area=int(motion_data.get("min_contour_area", motion_defaults.min_contour_area)),
    )

    evidence_data = settings.get("evidence") or {}
    evidence_defaults = EvidenceConfig()
    evidence = EvidenceConfig(
        root_dir=str(evidence_data.get("root_dir", evidence_defaults.root_dir)),
        pre_seconds=float(evidence_data.get("pre_seconds", evidence_defaults.pre_seconds)),
        post_seconds=float(evidence_data.get("post_seconds", evidence_defaults.post_seconds)),
        output_fps=float(evidence_data.get("output_fps", evidence_defaults.output_fps)),
    )

    raw_alerts = settings.get("alerts") or []
    alerts = [
        AlertStep(
            run=str(step["run"]),
            async_step=bool(step.get("async", False)),
            args=dict(step.get("args", {})),
        )
        for step in raw_alerts
    ]

    return AppConfig(streams=streams, motion=motion, evidence=evidence, alerts=alerts)


def resolve_root(root_dir: str) -> Path:
    path = Path(root_dir)
    if path.is_absolute():
        return path
    return Path.cwd() / path
