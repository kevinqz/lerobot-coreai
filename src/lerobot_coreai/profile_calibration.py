# profile_calibration.py — calibrate a safety profile from observed actions (v0.9.1).
#
# Fits SOFTWARE action bounds to actions observed in sim/shadow logs. It proves
# nothing about future actions or physical safety — only how a profile fits the
# actions it was calibrated on.

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .metrics import action_to_flat_float_list, infer_shape
from .safety_profiles import SafetyProfile

CALIBRATION_REPORT_SCHEMA_VERSION = "lerobot-coreai.profile_calibration_report.v0"

# Minimum bound clamps — never calibrate below these.
_MIN_ABS = 0.05
_MIN_DELTA = 0.01
_MIN_L2 = 0.05


@dataclass
class ProfileCalibrationConfig:
    actions_path: Path
    output_dir: Path
    base_profile: SafetyProfile | None = None
    output_profile: Path | None = None
    quantile: float = 0.995
    margin: float = 0.10
    min_samples: int = 10
    conservative: bool = False
    calibrate_max_abs: bool = True
    calibrate_max_delta: bool = True
    calibrate_max_l2_norm: bool = True
    preserve_robot_type: bool = True
    preserve_action_shape: bool = True
    name: str | None = None


@dataclass
class ProfileCalibrationResult:
    ok: bool
    samples: int
    profile: SafetyProfile
    report: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


_QUANTILE_KEYS = {0.50: "p50", 0.95: "p95", 0.99: "p99", 0.995: "p995"}


def _quantile_key(q: float) -> str:
    """Map a supported quantile to its statistics key (fail-closed on others)."""
    rounded = round(float(q), 3)
    if rounded in _QUANTILE_KEYS:
        return _QUANTILE_KEYS[rounded]
    raise CoreAIPolicyError(
        f"Unsupported calibration quantile: {q}. "
        "Supported: 0.50, 0.95, 0.99, 0.995."
    )


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def read_action_samples(actions_path: Path) -> dict[str, Any]:
    """Read actions.jsonl and return flats/shapes plus validity counts."""
    actions_path = Path(actions_path)
    if not actions_path.is_file():
        raise CoreAIPolicyError(f"Actions file not found: {actions_path}")
    flats: list[list[float]] = []
    shapes: list[tuple[int, ...]] = []
    valid = 0
    invalid = 0
    nan_steps = 0
    inf_steps = 0
    for line in actions_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        action = rec.get("action") if isinstance(rec, dict) else rec
        if action is None:
            invalid += 1
            continue
        try:
            flat = action_to_flat_float_list(action)
        except Exception:
            invalid += 1
            continue
        shape = infer_shape(action)
        has_nan = any(math.isnan(v) for v in flat)
        has_inf = any(math.isinf(v) for v in flat)
        if has_nan:
            nan_steps += 1
        if has_inf:
            inf_steps += 1
        valid += 1
        flats.append(flat)
        shapes.append(tuple(shape) if shape is not None else None)
    return {
        "flats": flats, "shapes": shapes, "valid": valid, "invalid": invalid,
        "nan_steps": nan_steps, "inf_steps": inf_steps,
    }


def compute_action_statistics(actions_path: Path) -> dict[str, Any]:
    """Compute abs/delta/L2 quantiles and shape stats from an actions file."""
    data = read_action_samples(actions_path)
    flats: list[list[float]] = data["flats"]

    abs_vals: list[float] = []
    l2_vals: list[float] = []
    delta_vals: list[float] = []
    prev: list[float] | None = None
    shape_changes = 0
    for flat in flats:
        finite = [v for v in flat if math.isfinite(v)]
        abs_vals.extend(abs(v) for v in finite)
        l2_vals.append(math.sqrt(sum(v * v for v in finite)))
        if prev is not None:
            if len(prev) == len(flat):
                delta_vals.append(max((abs(flat[i] - prev[i]) for i in range(len(flat))),
                                      default=0.0))
            else:
                shape_changes += 1
        prev = flat

    # Dominant / unique shapes.
    shape_counts: dict[tuple[int, ...], int] = {}
    for s in data["shapes"]:
        if s is not None:
            shape_counts[s] = shape_counts.get(s, 0) + 1
    dominant = max(shape_counts.items(), key=lambda kv: kv[1])[0] if shape_counts else None
    unique = [list(s) for s in sorted(shape_counts.keys())]

    def _q(vals):
        return {
            "p50": _percentile(vals, 0.50), "p95": _percentile(vals, 0.95),
            "p99": _percentile(vals, 0.99), "p995": _percentile(vals, 0.995),
            "max": max(vals) if vals else None,
        }

    return {
        "valid_actions": data["valid"],
        "invalid_actions": data["invalid"],
        "nan_action_steps": data["nan_steps"],
        "inf_action_steps": data["inf_steps"],
        "shape_changes": shape_changes,
        "dominant_shape": list(dominant) if dominant else None,
        "unique_shapes": unique,
        "abs": _q(abs_vals),
        "delta": _q(delta_vals),
        "l2_norm": _q(l2_vals),
    }


def calibrate_profile(config: ProfileCalibrationConfig) -> ProfileCalibrationResult:
    """Calibrate a software safety profile from observed actions."""
    stats = compute_action_statistics(config.actions_path)
    warnings: list[str] = []
    samples = stats["valid_actions"]

    if samples < config.min_samples:
        raise CoreAIPolicyError(
            f"Insufficient samples for calibration: {samples} < {config.min_samples}."
        )
    if stats["nan_action_steps"] or stats["inf_action_steps"]:
        warnings.append(
            f"non-finite actions observed: nan={stats['nan_action_steps']} "
            f"inf={stats['inf_action_steps']}"
        )

    base = config.base_profile
    m = 1.0 + config.margin
    quantile_key = _quantile_key(config.quantile)  # fail-closed on unsupported q

    def _bound(qkey: str, minimum: float, base_val):
        q = stats[qkey][quantile_key]
        if q is None:
            return base_val
        rec = max(q * m, minimum)
        if base_val is not None and rec > base_val:
            warnings.append(
                f"recommended {qkey} bound {rec:.4f} exceeds base profile {base_val}")
        if config.conservative and base_val is not None:
            rec = min(rec, base_val)
        return round(rec, 6)

    max_abs = _bound("abs", _MIN_ABS, getattr(base, "max_abs_action", None)) \
        if config.calibrate_max_abs else getattr(base, "max_abs_action", None)
    max_delta = _bound("delta", _MIN_DELTA, getattr(base, "max_delta", None)) \
        if config.calibrate_max_delta else getattr(base, "max_delta", None)
    max_l2 = _bound("l2_norm", _MIN_L2, getattr(base, "max_l2_norm", None)) \
        if config.calibrate_max_l2_norm else getattr(base, "max_l2_norm", None)

    robot_type = getattr(base, "robot_type", None) if config.preserve_robot_type else None
    action_shape = stats["dominant_shape"] if config.preserve_action_shape else None

    name = config.name or (
        f"{base.name}-calibrated" if base is not None else "calibrated-profile")
    method = f"quantile_{config.quantile}_margin_{config.margin}"
    if config.conservative:
        method += "_conservative"

    profile = SafetyProfile(
        name=name,
        profile_type="software_bounds",
        robot_type=robot_type,
        action_shape=action_shape,
        max_abs_action=max_abs,
        max_delta=max_delta,
        max_l2_norm=max_l2,
        require_finite=True,
        require_known_shape=action_shape is not None,
        require_robot_type_match=robot_type is not None,
        allow_shape_change=False,
        mode="fail_closed",
        calibrated_from=str(config.actions_path),
        calibration_method=method,
        limitations=[
            "Calibrated from observed simulator/shadow actions.",
            "May under-cover unseen behavior.",
            "Does not prove physical robot safety.",
            "Does not prove real-world safety.",
        ],
    )

    report = build_calibration_report(config, stats, profile, warnings)
    return ProfileCalibrationResult(
        ok=True, samples=samples, profile=profile, report=report, warnings=warnings,
    )


def build_calibration_report(
    config: ProfileCalibrationConfig, stats: dict[str, Any],
    profile: SafetyProfile, warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": True,
        "actions_path": str(config.actions_path),
        "samples": stats["valid_actions"],
        "valid_actions": stats["valid_actions"],
        "invalid_actions": stats["invalid_actions"],
        "nan_action_steps": stats["nan_action_steps"],
        "inf_action_steps": stats["inf_action_steps"],
        "dominant_shape": stats["dominant_shape"],
        "unique_shapes": stats["unique_shapes"],
        "statistics": {
            "abs": stats["abs"], "delta": stats["delta"], "l2_norm": stats["l2_norm"],
        },
        "recommended_bounds": {
            "max_abs_action": profile.max_abs_action,
            "max_delta": profile.max_delta,
            "max_l2_norm": profile.max_l2_norm,
        },
        "calibration_method": profile.calibration_method,
        "quantile": config.quantile,
        "quantile_key": _quantile_key(config.quantile),
        "margin": config.margin,
        "conservative": config.conservative,
        "warnings": warnings,
        "claims": {
            "proves_profile_fit_to_observed_actions": True,
            "proves_future_action_safety": False,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "proves_real_task_success": False,
        },
    }


def build_calibration_markdown(report: dict[str, Any]) -> str:
    rb = report.get("recommended_bounds", {})
    warns = report.get("warnings") or []
    warn_lines = "\n".join(f"- {w}" for w in warns) or "- None"
    return (
        "# Profile Calibration Report\n\n"
        f"- Actions: {report.get('actions_path')}\n"
        f"- Samples: {report.get('samples')}\n"
        f"- Dominant shape: {report.get('dominant_shape')}\n"
        f"- Method: {report.get('calibration_method')}\n\n"
        "## Recommended bounds\n\n"
        f"- max_abs_action: {rb.get('max_abs_action')}\n"
        f"- max_delta: {rb.get('max_delta')}\n"
        f"- max_l2_norm: {rb.get('max_l2_norm')}\n\n"
        "## Warnings\n\n"
        f"{warn_lines}\n\n"
        "## Claims\n\n"
        "This report fits a software profile to observed actions. "
        "It does not prove future action safety. "
        "It does not prove physical robot safety.\n"
    )
