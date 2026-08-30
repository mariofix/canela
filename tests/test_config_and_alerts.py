from __future__ import annotations

import json
from pathlib import Path

import pytest

from canela.alerts import AlertPipeline
from canela.config import AlertStep, load_config


@pytest.mark.asyncio
async def test_alert_pipeline_executes_python_callable(tmp_path: Path) -> None:
    marker = tmp_path / "marker.json"
    payload = {"stream": "cam-1"}

    step = AlertStep(
        run="tests.test_helpers::record_payload",
        args={"output_path": str(marker)},
    )

    pipeline = AlertPipeline([step])
    await pipeline.run(payload)

    assert marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8")) == payload


@pytest.mark.asyncio
async def test_alert_pipeline_executes_shell_command(tmp_path: Path) -> None:
    marker = tmp_path / "shell.txt"
    step = AlertStep(run=f"printf '%s' \"$CANELA_ALERT_PAYLOAD\" > {marker}")

    pipeline = AlertPipeline([step])
    await pipeline.run({"ok": True})

    assert marker.exists()
    assert json.loads(marker.read_text(encoding="utf-8")) == {"ok": True}


@pytest.mark.asyncio
async def test_alert_pipeline_raises_on_failing_shell_command() -> None:
    pipeline = AlertPipeline([AlertStep(run="exit 7")])

    with pytest.raises(RuntimeError, match="alert command failed"):
        await pipeline.run({"ok": True})


def test_load_config_with_multiple_resolutions_and_alerts(tmp_path: Path) -> None:
    settings = tmp_path / "settings.toml"
    settings.write_text(
        """
[default]

[[default.streams]]
name = "cam-1"
source = "rtsp://camera/one"
fps = 6
  [[default.streams.resolutions]]
  width = 640
  height = 360
  [[default.streams.resolutions]]
  width = 320
  height = 180

[[default.alerts]]
run = "pkg.module::fn"
async = true

[default.evidence]
pre_seconds = 7
post_seconds = 8

[default.motion]
warmup_frames = 45
""",
        encoding="utf-8",
    )

    config = load_config([str(settings)])

    assert len(config.streams) == 1
    assert len(config.streams[0].resolutions) == 2
    assert config.evidence.pre_seconds == 7
    assert config.evidence.post_seconds == 8
    assert config.motion.warmup_frames == 45
    assert config.alerts[0].run == "pkg.module::fn"
    assert config.alerts[0].async_step is True
