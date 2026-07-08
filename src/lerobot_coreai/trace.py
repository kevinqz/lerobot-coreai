# trace.py — JSONL trace writer for rollout events (v0.3).

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TraceWriter:
    """Writes rollout events as JSONL (one JSON object per line).

    Each line has the shape::

        {"ts": "2026-07-08T00:00:00Z", "event": "rollout.started", "data": {}}
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a")

    def write(self, event: str, data: dict[str, Any] | None = None) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "event": event,
            "data": data or {},
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
