# action_queue.py — per-timestep action queue over a chunk (v1.2.5).
#
# LeRobot's select_action returns ONE action per call and internally queues a
# predicted chunk. This queue models that: it accepts a chunk [H, A], pops one
# action [A] per step, and refuses ragged or non-finite chunks. The queue is
# owned by the Python bridge; reset() clears it. No hardware, no egress.

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .errors import CoreAIPolicyError


def _is_finite_row(row: Any) -> bool:
    try:
        return all(math.isfinite(float(x)) for x in row)
    except (TypeError, ValueError):
        return False


class ActionQueue:
    """A FIFO of per-timestep actions filled from a chunk."""

    def __init__(self) -> None:
        self._q: deque[list[float]] = deque()

    def __len__(self) -> int:
        return len(self._q)

    @property
    def empty(self) -> bool:
        return not self._q

    def load_chunk(self, chunk: Any) -> None:
        """Load a chunk [H, A]. Rejects ragged or non-finite chunks."""
        if not isinstance(chunk, (list, tuple)) or len(chunk) == 0:
            raise CoreAIPolicyError("action chunk must be a non-empty [H, A] sequence.")
        rows = [list(r) for r in chunk]
        width = len(rows[0])
        if width == 0:
            raise CoreAIPolicyError("action chunk rows must be non-empty.")
        for r in rows:
            if len(r) != width:
                raise CoreAIPolicyError(
                    f"ragged action chunk: row widths differ ({width} vs {len(r)}).")
            if not _is_finite_row(r):
                raise CoreAIPolicyError("action chunk contains non-finite values.")
        self._q.extend(rows)

    def pop_next(self) -> list[float]:
        """Pop the next per-timestep action [A]. Raises on exhaustion."""
        if not self._q:
            raise CoreAIPolicyError(
                "action queue exhausted; refill from a new chunk before popping.")
        return self._q.popleft()

    def reset(self) -> None:
        self._q.clear()
