from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dynaconf import Dynaconf


@dataclass(slots=True)
class Resolution:
    width: int
    height: int


@dataclass(slots=True)
class StreamConfig:
    name: str
    source: str
    fps: float = 5.0
    resolutions: list[Resolution] = field(default_factory=lambda: [Resolution(640, 360)])


@dataclass(slots=True)
class MotionConfig:
    delta_threshold: float = 20.0
    motion_ratio_threshold: float = 0.02
    cooldown_seconds: float = 10.0
    warmup_frames: int = 30


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
    return Resolution(width=int(value["width"]), height=int(value["height"]))


def load_config(settings_files: list[str] | None = None) -> AppConfig:
    files = settings_files or ["settings.toml", ".secrets.toml"]
    settings = Dynaconf(settings_files=files, environments=True)

    raw_streams = settings.get("streams") or []
    streams: list[StreamConfig] = []
    for item in raw_streams:
        name = str(item["name"])
        source = str(item["source"])
        fps = float(item.get("fps", 5.0))
        raw_resolutions = item.get("resolutions") or [{"width": 640, "height": 360}]
        resolutions = [_as_resolution(entry) for entry in raw_resolutions]
        streams.append(StreamConfig(name=name, source=source, fps=fps, resolutions=resolutions))

    motion_data = settings.get("motion") or {}
    motion_defaults = MotionConfig()
    motion = MotionConfig(
        delta_threshold=float(motion_data.get("delta_threshold", motion_defaults.delta_threshold)),
        motion_ratio_threshold=float(
            motion_data.get("motion_ratio_threshold", motion_defaults.motion_ratio_threshold)
        ),
        cooldown_seconds=float(motion_data.get("cooldown_seconds", motion_defaults.cooldown_seconds)),
        warmup_frames=int(motion_data.get("warmup_frames", motion_defaults.warmup_frames)),
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
