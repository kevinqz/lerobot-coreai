# action_validation.py — strict Runner-output normalization (v1.3.3).
#
# The Runner's action output must be validated before it becomes a LeRobot
# tensor — a ragged, rank-4, non-finite, or wrong-dimension chunk must be
# rejected, not silently reshaped. This is the single normalizer both
# predict_action_chunk and select_action route through. No hardware, no egress.

from __future__ import annotations

import math
from typing import Any

import torch

from lerobot_coreai.errors import CoreAIPolicyError


def _all_finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def normalize_and_validate_action_chunk(
    raw_action: Any, *, representation: str, expected_batch_size: int = 1,
    expected_horizon: int | None = None, expected_action_dim: int | None = None,
    device: Any = None,
) -> torch.Tensor:
    """Normalize a Runner action to a validated ``(B, H, A)`` float tensor.

    Permitted shapes for B=1:
      single: ``[A]`` or ``[1, A]``      -> ``[1, 1, A]``
      chunk:  ``[H, A]`` or ``[1, H, A]`` -> ``[1, H, A]``
    Anything else (ragged, rank>3, wrong dim/horizon, non-finite) fails closed.
    """
    try:
        t = torch.as_tensor(raw_action, dtype=torch.float32)
    except Exception as e:
        raise CoreAIPolicyError(f"action is not a numeric tensor: {e}")
    if device is not None:
        t = t.to(device)

    if representation == "single":
        if t.ndim == 1:
            t = t.unsqueeze(0).unsqueeze(0)     # [A] -> [1,1,A]
        elif t.ndim == 2 and t.shape[0] == 1:
            t = t.unsqueeze(1)                  # [1,A] -> [1,1,A]
        else:
            raise CoreAIPolicyError(
                f"single action must be [A] or [1,A]; got shape {tuple(t.shape)}.")
    else:  # chunk
        if t.ndim == 2:
            t = t.unsqueeze(0)                  # [H,A] -> [1,H,A]
        elif t.ndim == 3 and t.shape[0] == 1:
            pass                               # [1,H,A]
        else:
            raise CoreAIPolicyError(
                f"chunk action must be [H,A] or [1,H,A]; got shape {tuple(t.shape)}.")

    if t.ndim != 3:
        raise CoreAIPolicyError(f"normalized action must be rank 3; got {t.ndim}.")
    b, h, a = t.shape
    if b != expected_batch_size:
        raise CoreAIPolicyError(f"batch {b} != expected {expected_batch_size}.")
    if expected_horizon is not None and h != expected_horizon:
        raise CoreAIPolicyError(f"horizon {h} != expected {expected_horizon}.")
    if expected_action_dim is not None and a != expected_action_dim:
        raise CoreAIPolicyError(f"action_dim {a} != expected {expected_action_dim}.")
    if not _all_finite(t):
        raise CoreAIPolicyError("action contains non-finite values.")
    return t
