# reports.py — helpers for building rollout reports (v0.3).

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def infer_shape(value: Any) -> list[int] | None:
    """Infer the shape of a nested list. Returns None for ragged or non-list."""
    if isinstance(value, str):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return [0]
        child_shapes = [infer_shape(v) for v in value]
        first = child_shapes[0]
        if any(s != first for s in child_shapes):
            return None  # ragged
        if first is not None:
            return [len(value)] + first
        return [len(value)]
    return None


def contains_nan(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(contains_nan(v) for v in value)
    if isinstance(value, float):
        return math.isnan(value)
    return False


def contains_inf(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(contains_inf(v) for v in value)
    if isinstance(value, float):
        return math.isinf(value)
    return False


def build_success_report(
    *,
    policy_path: str,
    source_repo_id: str | None,
    policy_type: str,
    model_id: str,
    robot_type: str | None,
    runner_url: str,
    runner_timing: dict[str, float] | None,
    parity_passed: bool,
    fixture_source: str,
    observation_keys: list[str],
    action: Any,
    files: dict[str, str],
) -> dict[str, Any]:
    action_shape = infer_shape(action)
    return {
        "schema_version": "lerobot-coreai.rollout_report.v0",
        "lerobot_coreai_version": __version__,
        "ok": True,
        "mode": "dry_run",
        "policy": {
            "path": policy_path,
            "repo_id": policy_path,
            "source_repo_id": source_repo_id,
            "type": policy_type,
            "runtime": "coreai",
            "model_id": model_id,
        },
        "robot": {
            "type": robot_type,
            "connected": False,
            "actions_sent": 0,
        },
        "runner": {
            "url": runner_url,
            "reachable": True,
            "supports_action": True,
            "timing": runner_timing or {},
        },
        "manifest": {
            "parity_passed": parity_passed,
            "default_mode": "dry_run",
        },
        "observation": {
            "source": fixture_source,
            "frames": 1,
            "features_valid": True,
            "keys": observation_keys,
        },
        "action": {
            "generated": True,
            "shape": action_shape,
            "contains_nan": contains_nan(action),
            "contains_inf": contains_inf(action),
        },
        "safety": {
            "physical_actuation_possible": False,
            "mode": "dry_run",
            "confirmation_required_for_real": True,
            "confirmation_provided": False,
            "motor_commands_available": False,
        },
        "files": files,
        "errors": [],
    }


def build_failure_report(
    *,
    policy_path: str | None,
    robot_type: str | None,
    mode: str,
    error_type: str,
    error_message: str,
    stage: str,
    runner_url: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "lerobot-coreai.rollout_report.v0",
        "lerobot_coreai_version": __version__,
        "ok": False,
        "mode": mode,
        "policy": {"path": policy_path} if policy_path else {},
        "robot": {
            "type": robot_type,
            "connected": False,
            "actions_sent": 0,
        },
        "runner": {"url": runner_url, "reachable": False} if runner_url else {},
        "safety": {
            "physical_actuation_possible": False,
            "motor_commands_available": False,
        },
        "errors": [
            {
                "type": error_type,
                "message": error_message,
                "stage": stage,
                "recoverable": True,
            }
        ],
    }
