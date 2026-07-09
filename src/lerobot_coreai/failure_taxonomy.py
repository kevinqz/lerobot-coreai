# failure_taxonomy.py — classify sim run errors into a taxonomy (v0.8.2).
#
# Turns the flat errors list from a sim run into a structured taxonomy:
# counts by stage, by type, a coarse classification (runner/environment/
# validation), and the first failure for quick triage.
#
# This is a diagnostic aid, not a safety gate.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FailureRecord:
    """A single classified failure."""

    stage: str
    type: str
    message: str
    episode: int | None = None
    step: int | None = None


# Known stages that appear in sim run error records.
_KNOWN_STAGES = {
    "observation_adapt",
    "observation_serialize",
    "action_generate",
    "simulator_step",
    "environment_reset",
    "environment_build",
    "policy_load",
    "robot_type_validation",
    "loop",
}


def classify_failure(error: dict[str, Any]) -> str:
    """Coarsely classify an error record into a failure category.

    Returns one of: runner, environment, validation, unknown.
    """
    stage = (error.get("stage") or "").lower()
    etype = (error.get("type") or "").lower()

    if "validation" in etype or "observation" in stage and "adapt" not in stage:
        return "validation"
    if stage.startswith("action.generate") or "runner" in etype or stage == "policy_load":
        return "runner"
    if any(k in stage for k in ("env", "simulator", "environment")):
        return "environment"
    return "unknown"


def build_failure_taxonomy(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a structured failure taxonomy from an errors list.

    Output:
        {
          "total_failures": int,
          "by_stage": {stage: count},
          "by_type": {type: count},
          "classified": {category: count},
          "first_failure": {...} | None,
        }
    """
    by_stage: dict[str, int] = {}
    by_type: dict[str, int] = {}
    classified: dict[str, int] = {"runner": 0, "environment": 0, "validation": 0, "unknown": 0}
    first_failure: dict[str, Any] | None = None

    for err in errors:
        stage = err.get("stage", "unknown")
        etype = err.get("type", "unknown")
        by_stage[stage] = by_stage.get(stage, 0) + 1
        by_type[etype] = by_type.get(etype, 0) + 1

        category = classify_failure(err)
        classified[category] = classified.get(category, 0) + 1

        if first_failure is None:
            first_failure = {
                "episode": err.get("episode"),
                "step": err.get("step"),
                "stage": stage,
                "type": etype,
                "message": err.get("message"),
            }

    return {
        "total_failures": len(errors),
        "by_stage": by_stage,
        "by_type": by_type,
        "classified": classified,
        "first_failure": first_failure,
    }
