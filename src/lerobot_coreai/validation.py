# validation.py — manifest-based validation for observation/action (v0.2).
#
# This is the safety/quality layer: ensures the batch matches the policy's feature
# contract before calling the runner, and ensures the action output is well-formed.
# No clipping or modification — validation only.

from __future__ import annotations

import math
from typing import Any

from .errors import ActionValidationError, ObservationValidationError
from .manifest import LeRobotCoreAIManifest


def validate_observation_batch(
    batch: dict[str, Any],
    manifest: LeRobotCoreAIManifest,
    *,
    strict_observation_keys: bool = False,
) -> None:
    """Validate that the observation batch matches the manifest's feature contract.

    Checks:
    - All required observation keys are present.
    - Optional keys may be absent.
    - Unknown keys are ignored unless strict_observation_keys=True.
    - observation.state (float32) has the expected shape if declared.
    - task is a string if present.

    Raises:
        ObservationValidationError: On any mismatch.
    """
    # Check required keys.
    missing = [
        name for name, spec in manifest.observation_features.items()
        if spec.required and name not in batch
    ]
    if missing:
        raise ObservationValidationError(
            f"Missing required observation keys: {', '.join(missing)}.\n"
            f"Expected keys: {_format_expected(manifest)}"
        )

    # Check unknown keys.
    known_keys = set(manifest.observation_features.keys())
    batch_keys = set(batch.keys())
    unknown = batch_keys - known_keys
    if unknown and strict_observation_keys:
        raise ObservationValidationError(
            f"Unknown observation keys (strict mode): {', '.join(sorted(unknown))}.\n"
            f"Allowed keys: {', '.join(sorted(known_keys))}"
        )

    # Validate state tensor shape.
    state_spec = manifest.observation_features.get("observation.state")
    if state_spec and state_spec.shape and "observation.state" in batch:
        value = batch["observation.state"]
        actual_shape = _get_shape(value)
        if actual_shape is not None and actual_shape != list(state_spec.shape):
            raise ObservationValidationError(
                f"observation.state shape mismatch: expected {state_spec.shape}, got {actual_shape}"
            )

    # Validate task is string.
    if "task" in batch and batch["task"] is not None:
        if not isinstance(batch["task"], str):
            raise ObservationValidationError(
                f"observation 'task' must be a string, got {type(batch['task']).__name__}"
            )


def validate_action_output(
    action: Any,
    manifest: LeRobotCoreAIManifest,
) -> None:
    """Validate that the action output from the runner is well-formed.

    Checks:
    - Action exists and is a list/nested list.
    - Shape matches the manifest's action features.
    - No NaN values.
    - No Inf values.

    Raises:
        ActionValidationError: On any mismatch.
    """
    if action is None:
        raise ActionValidationError("Action is None — runner returned no action.")

    if not isinstance(action, (list, tuple)):
        raise ActionValidationError(
            f"Action must be a list of lists, got {type(action).__name__}"
        )

    # Check shape if declared in manifest.
    action_spec = manifest.action_features.get("action")
    if action_spec and action_spec.shape:
        expected = action_spec.shape  # e.g. [16, 7]
        actual = _get_nested_shape(action)
        if actual is not None and actual != list(expected):
            raise ActionValidationError(
                f"Action shape mismatch: expected {expected}, got {actual}"
            )

    # Check for NaN/Inf.
    bad = _find_nan_inf(action)
    if bad:
        kind, index = bad
        raise ActionValidationError(
            f"Action contains {kind} at index {index}"
        )


def validate_robot_type(
    requested_robot_type: str | None,
    manifest: LeRobotCoreAIManifest,
) -> None:
    """Validate that the requested robot type matches the manifest.

    Raises:
        ObservationValidationError: On mismatch.
    """
    if requested_robot_type is None:
        return  # No robot type requested — skip.
    if requested_robot_type != manifest.robot_type:
        raise ObservationValidationError(
            f"Robot type mismatch: policy expects '{manifest.robot_type}', "
            f"got '{requested_robot_type}'"
        )


# --- Helpers ---

def _format_expected(manifest: LeRobotCoreAIManifest) -> str:
    parts = []
    for name, spec in manifest.observation_features.items():
        shape_str = f" {list(spec.shape)}" if spec.shape else ""
        req = "required" if spec.required else "optional"
        parts.append(f"  - {name}: {spec.dtype}{shape_str} ({req})")
    return "\n" + "\n".join(parts)


def _get_shape(value: Any) -> list[int] | None:
    """Get the shape of a list or nested list. Returns None for scalars/strings."""
    if isinstance(value, str):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [0]
        inner = _get_shape(value[0])
        if inner is not None:
            return [len(value)] + inner
        return [len(value)]
    return None


def _get_nested_shape(value: Any) -> list[int] | None:
    """Get the full nested shape of a list of lists."""
    return _get_shape(value)


def _find_nan_inf(action: Any, index: list[int] | None = None) -> tuple[str, str] | None:
    """Recursively check for NaN/Inf in a nested list. Returns (kind, index_str) or None."""
    if index is None:
        index = []
    if isinstance(action, (list, tuple)):
        for i, item in enumerate(action):
            result = _find_nan_inf(item, index + [i])
            if result:
                return result
    elif isinstance(action, float):
        if math.isnan(action):
            return ("NaN", str(index))
        if math.isinf(action):
            return ("Inf", str(index))
    return None
