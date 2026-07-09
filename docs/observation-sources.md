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

### `camera` — live camera (experimental, stub)

Camera capture is **experimental** and will be available in v0.7.1. In v0.7.0, passing
`--observation-source camera` raises `CoreAIPolicyError` with a message directing you to
use `folder` or `fixtures`.

OpenCV will **not** be a hard dependency when camera support lands — it will be an optional
extra (`[camera]`). Shadow mode works fully without it.

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
