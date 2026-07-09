# profile_recommendation.py — heuristic built-in profile recommendation (v0.9.1).
#
# Recommends a built-in software safety profile from a policy manifest and/or the
# dominant action shape observed in an actions log. Heuristic only — it does not
# prove physical safety.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProfileRecommendation:
    recommended_profile: str
    confidence: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_profile": self.recommended_profile,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "warnings": self.warnings,
        }


_WARNING = "Recommendation is heuristic and does not prove physical safety."


def recommend_profile(
    *, robot_type: str | None = None, dominant_shape: list[int] | None = None,
    env_id: str | None = None, policy_type: str | None = None,
) -> ProfileRecommendation:
    """Recommend a built-in profile from available signals."""
    reasons: list[str] = []

    # 1. Robot type is the strongest signal.
    if robot_type == "so100":
        reasons.append("policy manifest robot_type=so100")
        return ProfileRecommendation("so100-sim-default", "high", reasons, [_WARNING])
    if robot_type == "so101":
        reasons.append("policy manifest robot_type=so101")
        return ProfileRecommendation("so101-sim-default", "high", reasons, [_WARNING])

    # 2. Environment hint.
    if env_id and "pusht" in env_id.lower():
        reasons.append(f"env_id={env_id}")
        return ProfileRecommendation("pusht-sim-default", "medium", reasons, [_WARNING])

    # 3. Dominant action shape.
    if dominant_shape is not None:
        reasons.append(f"dominant action shape={dominant_shape}")
        if dominant_shape == [2]:
            return ProfileRecommendation("pusht-sim-default", "medium", reasons, [_WARNING])
        if dominant_shape == [16, 7]:
            return ProfileRecommendation("so100-sim-default", "medium", reasons, [_WARNING])
        if dominant_shape == [7]:
            return ProfileRecommendation("generic-7dof-sim-default", "medium", reasons, [_WARNING])

    # 4. Fallback.
    reasons.append("no strong robot/shape signal")
    return ProfileRecommendation(
        "default-sim-safe", "low", reasons,
        [_WARNING, "Falling back to the generic conservative default profile."],
    )


def recommend_from_actions(
    actions_path: Path, *, robot_type: str | None = None,
    env_id: str | None = None, policy_type: str | None = None,
) -> ProfileRecommendation:
    """Recommend a profile using the dominant shape from an actions log."""
    from .profile_calibration import compute_action_statistics
    dominant = None
    try:
        dominant = compute_action_statistics(actions_path)["dominant_shape"]
    except Exception:
        dominant = None
    return recommend_profile(
        robot_type=robot_type, dominant_shape=dominant,
        env_id=env_id, policy_type=policy_type,
    )
