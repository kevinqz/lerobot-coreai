# safety_profiles.py — safety profiles for the runtime supervisor (v0.9.0).
#
# A SafetyProfile is a conservative *software* contract for action bounds. It is
# NOT a certified hardware safety envelope and does not prove physical robot
# safety. Profiles are loaded from a JSON file or from a built-in name.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import CoreAIPolicyError

SAFETY_PROFILE_SCHEMA_VERSION = "lerobot-coreai.safety_profile.v0"


@dataclass
class SafetyProfile:
    """Static, conservative software bounds for a family of robot/policy/env.

    A profile never proves physical safety — it only bounds what a software
    supervisor will let egress. `mode` is fixed to "fail_closed": on any
    uncertain critical condition, the supervisor blocks.
    """

    name: str
    robot_type: str | None = None
    action_shape: list[int] | None = None
    min_action: float | list[float] | None = None
    max_action: float | list[float] | None = None
    max_abs_action: float | None = None
    max_delta: float | None = None
    max_l2_norm: float | None = None
    allow_nan: bool = False
    allow_inf: bool = False
    allow_shape_change: bool = False
    clip_to_bounds: bool = False
    block_on_clip: bool = False
    require_finite: bool = True
    require_known_shape: bool = True
    require_robot_type_match: bool = True
    mode: str = "fail_closed"
    description: str | None = None
    source: str | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SAFETY_PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "robot_type": self.robot_type,
            "action_shape": self.action_shape,
            "min_action": self.min_action,
            "max_action": self.max_action,
            "max_abs_action": self.max_abs_action,
            "max_delta": self.max_delta,
            "max_l2_norm": self.max_l2_norm,
            "allow_nan": self.allow_nan,
            "allow_inf": self.allow_inf,
            "allow_shape_change": self.allow_shape_change,
            "clip_to_bounds": self.clip_to_bounds,
            "block_on_clip": self.block_on_clip,
            "require_finite": self.require_finite,
            "require_known_shape": self.require_known_shape,
            "require_robot_type_match": self.require_robot_type_match,
            "mode": self.mode,
            "description": self.description,
        }


_ALLOWED_KEYS = {
    "name", "robot_type", "action_shape", "min_action", "max_action",
    "max_abs_action", "max_delta", "max_l2_norm", "allow_nan", "allow_inf",
    "allow_shape_change", "clip_to_bounds", "block_on_clip", "require_finite",
    "require_known_shape", "require_robot_type_match", "mode", "description",
}


def _load_schema() -> dict[str, Any]:
    return json.loads(
        files("lerobot_coreai.schemas").joinpath("safety-profile.schema.json").read_text()
    )


def profile_from_dict(data: dict[str, Any], *, source: str | None = None) -> SafetyProfile:
    """Build a SafetyProfile from a validated dict."""
    if data.get("mode", "fail_closed") != "fail_closed":
        raise CoreAIPolicyError(
            f"Safety profile mode must be 'fail_closed', got {data.get('mode')!r}."
        )
    kwargs = {k: v for k, v in data.items() if k in _ALLOWED_KEYS}
    if "name" not in kwargs:
        raise CoreAIPolicyError("Safety profile is missing required field 'name'.")
    return SafetyProfile(source=source, **kwargs)


def load_safety_profile(path: Path) -> SafetyProfile:
    """Load and validate a safety profile from a JSON file."""
    path = Path(path)
    if not path.is_file():
        raise CoreAIPolicyError(f"Safety profile not found: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise CoreAIPolicyError(f"Invalid safety profile JSON: {e}") from None
    _validate_profile_dict(data)
    return profile_from_dict(data, source=str(path))


def load_builtin_profile(name: str) -> SafetyProfile:
    """Load a built-in safety profile by name (with or without .json)."""
    fname = name if name.endswith(".json") else f"{name}.json"
    try:
        text = files("lerobot_coreai.profiles").joinpath(fname).read_text()
    except (FileNotFoundError, ModuleNotFoundError):
        raise CoreAIPolicyError(f"Unknown built-in safety profile: {name}") from None
    data = json.loads(text)
    _validate_profile_dict(data)
    return profile_from_dict(data, source=f"builtin:{fname}")


def resolve_safety_profile(
    *, path: Path | None = None, name: str | None = None,
    default_builtin: str | None = "default-sim-safe",
) -> SafetyProfile:
    """Resolve a profile from an explicit path, a built-in name, or a default.

    Fail-closed: if nothing resolves and no default is given, raises.
    """
    if path is not None:
        return load_safety_profile(path)
    if name is not None:
        return load_builtin_profile(name)
    if default_builtin is not None:
        return load_builtin_profile(default_builtin)
    raise CoreAIPolicyError(
        "No safety profile specified and no default available (fail-closed)."
    )


def _validate_profile_dict(data: dict[str, Any]) -> None:
    try:
        import jsonschema
        jsonschema.validate(data, _load_schema())
    except Exception as e:
        message = getattr(e, "message", str(e))
        raise CoreAIPolicyError(f"Invalid safety profile: {message}") from None
