# observation_adapters.py — declarative observation adaptation for shadow mode (v0.7.2).
#
# Adapts raw observations from sources into the shape the policy/runner expects.
# Handles key mapping, state injection, task injection, required-key checks, and
# manifest-driven filtering. Returns warnings for non-fatal issues, raises for fatal ones.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .errors import CoreAIPolicyError

if TYPE_CHECKING:
    from .manifest import LeRobotCoreAIManifest


@dataclass
class ObservationAdapterConfig:
    """Configuration for observation adaptation."""

    image_key: str = "observation.images.wrist"
    image_keys: dict[str, str] | None = None  # alias → observation key mapping
    state_key: str = "observation.state"
    state_vector: list[float] | None = None
    state_json: Path | None = None
    task: str | None = None
    require_task: bool = False
    require_state: bool = False
    required_keys: list[str] = field(default_factory=list)
    drop_unknown_keys: bool = False


@dataclass
class AdaptedObservation:
    """Result of observation adaptation."""

    observation: dict[str, Any]
    keys_present: list[str]
    keys_missing: list[str]
    warnings: list[str]


def adapt_observation(
    raw: dict[str, Any],
    config: ObservationAdapterConfig,
    *,
    manifest: "LeRobotCoreAIManifest | None" = None,
) -> AdaptedObservation:
    """Adapt a raw observation into the shape the policy expects.

    Behavior:
    - Start from a copy of the raw observation.
    - Apply image key mapping (alias → canonical key) if image_keys provided.
    - Inject state_vector or state_json into state_key if provided.
    - Inject task if provided.
    - Check required_keys (config + manifest-required observation keys).
    - If drop_unknown_keys, keep only keys present in manifest observation_features.
    - Collect warnings for non-fatal issues.

    Raises:
        CoreAIPolicyError: If a required key is missing, state_json is invalid,
            or state_vector contains non-numeric values.
    """
    warnings: list[str] = []
    obs: dict[str, Any] = dict(raw)

    # Apply image key alias mapping.
    if config.image_keys:
        for alias, canonical in config.image_keys.items():
            if alias in obs and canonical != alias:
                obs[canonical] = obs.pop(alias)

    # Inject state.
    if config.state_vector is not None:
        for v in config.state_vector:
            try:
                float(v)
            except (TypeError, ValueError):
                raise CoreAIPolicyError(
                    f"state_vector contains non-numeric value: {v!r}"
                )
        obs[config.state_key] = list(config.state_vector)
    elif config.state_json is not None:
        try:
            data = json.loads(Path(config.state_json).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise CoreAIPolicyError(
                f"Failed to load state JSON from {config.state_json}: {e}"
            ) from e
        if not isinstance(data, list):
            raise CoreAIPolicyError(
                f"State JSON must be an array of numbers, got {type(data).__name__}"
            )
        obs[config.state_key] = [float(v) for v in data]

    # Inject task.
    if config.task is not None:
        obs["task"] = config.task

    # Determine required keys (only from explicit config, not manifest —
    # manifest validation is handled by policy.predict_action).
    required = list(config.required_keys)
    if config.require_task and "task" not in required:
        required.append("task")
    if config.require_state and config.state_key not in required:
        required.append(config.state_key)

    # When drop_unknown_keys is set, use manifest to know which keys to keep,
    # but still warn (not error) about missing manifest-required keys.
    if config.drop_unknown_keys and manifest is not None:
        for name, feat in manifest.observation_features.items():
            if getattr(feat, "required", False) and name not in obs:
                warnings.append(f"Manifest-required key not in observation: {name}")

    # Check required keys present.
    keys_present = [k for k in obs.keys()]
    keys_missing = [k for k in required if k not in obs]
    if keys_missing:
        raise CoreAIPolicyError(
            f"Required observation keys missing: {keys_missing}"
        )

    # Drop unknown keys (keep only manifest keys).
    if config.drop_unknown_keys and manifest is not None:
        manifest_keys = set(manifest.observation_features.keys())
        manifest_keys.add("task")  # task is always allowed
        dropped = [k for k in list(obs.keys()) if k not in manifest_keys]
        if dropped:
            warnings.append(f"Dropped non-manifest keys: {dropped}")
            for k in dropped:
                obs.pop(k)

    return AdaptedObservation(
        observation=obs,
        keys_present=sorted(obs.keys()),
        keys_missing=[],
        warnings=warnings,
    )
