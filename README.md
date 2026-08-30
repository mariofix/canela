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
  - Arbitrary Linux shell commands

## Configuration
Uses Dynaconf with `settings.toml` by default.

```toml
[default]
[[default.streams]]
name = "front-door"
source = "rtsp://camera.local/stream"
fps = 5
  [[default.streams.resolutions]]
  width = 640
  height = 360

[default.evidence]
pre_seconds = 5
post_seconds = 5

[[default.alerts]]
run = "awesome_lib.awesome_stuff::process"
async = false

[[default.alerts]]
run = "bash /opt/alerts/pager.sh"
async = true
```

## Run
```bash
python -m canela --settings settings.toml
```
