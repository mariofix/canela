import os
from dataclasses import dataclass

from dotenv import load_dotenv
from dynaconf import Dynaconf

load_dotenv()

settings = Dynaconf(
    envvar_prefix="CANELA",
    settings_files=["settings.toml"],
)


@dataclass
class StreamConfig:
    name: str
    rtsp_url: str
    save_dir: str
    detect_width: int
    detect_height: int
    min_contour_area: int
    cooldown: float
    warmup_frames: int
    history: int
    var_threshold: int
    audio_enabled: bool
    audio_rms_threshold: float
    audio_cooldown: float


def resolve_stream_config(name, rtsp_url, overrides=None):
    overrides = overrides or {}

    def get(key, default):
        return overrides.get(key, settings.get(key, default))

    return StreamConfig(
        name=name,
        rtsp_url=rtsp_url,
        save_dir=overrides.get("save_dir", os.path.join(settings.get("save_dir", "motion_captures"), name)),
        detect_width=get("detect_width", 640),
        detect_height=get("detect_height", 360),
        min_contour_area=get("min_contour_area", 500),
        cooldown=get("cooldown", 5.0),
        warmup_frames=get("warmup_frames", 30),
        history=get("history", 300),
        var_threshold=get("var_threshold", 32),
        audio_enabled=get("audio_enabled", False),
        audio_rms_threshold=get("audio_rms_threshold", 2500.0),
        audio_cooldown=get("audio_cooldown", 5.0),
    )


def build_stream_configs(args):
    if args.rtsp_url:
        # Ad-hoc mode: bypass settings.toml entirely, use one stream from CLI.
        return [resolve_stream_config(args.name, args.rtsp_url)]

    configured = settings.get("streams", [])

    if not configured:
        raise SystemExit("no streams configured in settings.toml and no --rtsp-url given")

    if args.only:
        wanted = set(args.only.split(","))
        configured = [s for s in configured if s.get("name") in wanted]

        if not configured:
            raise SystemExit(f"no configured streams match --only {args.only}")

    stream_configs = []
    for stream in configured:
        if not stream.get("rtsp_url"):
            raise SystemExit(f"stream '{stream.get('name')}' has no rtsp_url set (check .env)")

        stream_configs.append(resolve_stream_config(stream["name"], stream["rtsp_url"], overrides=stream))

    return stream_configs