# Observation Sources

Shadow mode reads observations from pluggable sources via the `ObservationSource` protocol.
This document describes the sources available in v0.7.

## Protocol

```python
class ObservationSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> dict[str, Any] | None: ...  # None = EOF
    def close(self) -> None: ...
```

`read()` returns `None` when the source is exhausted (EOF). The shadow loop stops
when it gets `None` or hits `--max-steps` / `--duration-seconds`.

## Sources

### `fixture` — single observation fixture

Reads one observation fixture JSON file. With `repeat=True` (the default for `--observation-source fixture`),
the same observation is yielded for every step. With `repeat=False`, it yields once then EOF.

**Fixture format:** See [fixture-format.md](fixture-format.md) for flat and typed fixture formats.

```bash
lerobot-coreai shadow \
  --observation-source fixture \
  --fixture examples/evo1_so100_observation.json \
  ...
```

### `fixtures` — ordered fixture directory

Reads a directory of JSON fixtures in lexicographic order:

```
fixtures/
  000000.json
  000001.json
  000002.json
```

EOF when the last file is read.

```bash
lerobot-coreai shadow \
  --observation-source fixtures \
  --fixtures-dir examples/shadow_sequence \
  ...
```

### `folder` — image frames from a directory

Reads image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.webp`) from a directory in sorted order.
Builds an observation batch per image:

```json
{
  "observation.images.wrist": "/abs/path/to/000000.png",
  "observation.state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
  "task": "pick up the cube"
}
```

- `--image-key` controls the observation key (default: `observation.images.wrist`)
- `--state-json` provides `observation.state` from a JSON array file
- `--state-vector` provides `observation.state` as comma-separated floats
- `--task` adds a `task` string to every observation

```bash
lerobot-coreai shadow \
  --observation-source folder \
  --frames-dir data/shadow_frames \
  --image-key observation.images.wrist \
  --state-vector 0.0,0.0,0.0,0.0,0.0,0.0,0.0 \
  --task "pick up the cube" \
  ...
```

### `camera` — live camera (experimental)

Captures frames from a local RGB camera via OpenCV, saves them to disk, and returns
an observation with the image key pointing to the saved frame path.

**Install:**
```bash
pip install "lerobot-coreai[camera]"
```

cv2 is imported lazily at `open()` time — the core package works without it. If cv2 is
not installed, a clear `CoreAIPolicyError` with install instructions is raised.

```bash
lerobot-coreai shadow \
  --observation-source camera \
  --camera.index 0 \
  --camera.width 1280 \
  --camera.height 720 \
  --camera.fps 10 \
  --task "pick up the cube" \
  --state-vector "0,0,0,0,0,0,0" \
  --output-dir runs/camera-shadow \
  ...
```

Camera args:
- `--camera.index` (default: 0) — camera device index
- `--camera.width` — requested frame width
- `--camera.height` — requested frame height
- `--camera.fps` — requested capture FPS
- `--no-save-camera-frames` — don't save frames to disk (default: frames saved to `frames/`)

Camera source is **observation-only**. It does not connect to a robot or actuator.
All actions generated during a camera shadow run are blocked by `ActionBlocker`.

## Factory

The `build_observation_source()` function dispatches by source type name:

```python
from lerobot_coreai.observation_sources import build_observation_source

source = build_observation_source(
    "folder",
    frames_dir=Path("data/shadow_frames"),
    image_key="observation.images.wrist",
    task="pick up the cube",
)
```

Raises `CoreAIPolicyError` if required arguments are missing or the type is unknown.
