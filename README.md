# canela
Private Security System for the Homestead

## Features
- Motion detection across multiple streams
- Multiple resolutions per stream for detection checks
- On motion, evidence folder with:
  - `motion.json` metadata
  - `snapshot.jpg`
  - `clip.mp4` containing configurable pre/post seconds
- Configurable alert pipeline with sequential and async steps
  - Python callables (`module.path::function`)
  - Arbitrary Linux shell commands (trusted operator-managed configuration)

## Configuration
Uses Dynaconf with `settings.toml` by default.

```toml
[default]
[[default.streams]]
name = "camera-model-a"
fps = 5
  [[default.streams.resolutions]]
  name = "high"
  source = "rtsp://127.0.0.1:554/s0"
  [[default.streams.resolutions]]
  name = "mid"
  source = "rtsp://127.0.0.1:554/s2"
  [[default.streams.resolutions]]
  name = "low"
  source = "rtsp://127.0.0.1:554/s1"

[default.evidence]
pre_seconds = 5
post_seconds = 5

[default.motion]
warmup_frames = 30

[[default.alerts]]
run = "awesome_lib.awesome_stuff::process"
async = false

[[default.alerts]]
run = "bash /opt/alerts/pager.sh"
async = true
```

You can still use legacy single-source configs (`stream.source` + width/height resolutions); when `resolutions[*].source` is present, each resolution is read from its own RTSP endpoint.

## Telegram alert example

To notify a Telegram chat with the motion snapshot and detection metadata, add an alert step that calls the bundled helper:

```toml
[[default.alerts]]
run = "canela.telegram_alert::send_telegram_alert"
async = false
args = { chat_id = "123456789", token = "<BOT_TOKEN>" }
```

You can also omit `chat_id` and `token` from `args` and set `TELEGRAM_CHAT_ID` / `TELEGRAM_BOT_TOKEN` in the environment instead.

## Run
```bash
poetry run canela --settings settings.toml
```

For more runtime details, increase verbosity:

```bash
poetry run canela --settings settings.toml --log-level DEBUG
```
