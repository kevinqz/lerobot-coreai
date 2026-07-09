# profile_reports.py — profile comparison + fit reports (v0.9.1).
#
# Runs two software safety profiles over the same actions log and reports how
# they differ. Proves nothing about physical safety or profile equivalence in
# general — only behavior on the provided actions.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import __version__
from .safety_profiles import SafetyProfile
from .safety_supervisor import MODE_ENFORCE, SafetyContext, SafetySupervisor, safe_evaluate

COMPARISON_REPORT_SCHEMA_VERSION = "lerobot-coreai.profile_comparison_report.v0"


def _iter_actions(actions_path: Path):
    for line in Path(actions_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            yield None
            continue
        yield rec.get("action") if isinstance(rec, dict) else rec


def compare_profiles(
    profile_a: SafetyProfile, profile_b: SafetyProfile, actions_path: Path,
) -> dict[str, Any]:
    """Run both profiles over the actions log and summarize the differences."""
    sup_a = SafetySupervisor(profile_a, mode=MODE_ENFORCE)
    sup_b = SafetySupervisor(profile_b, mode=MODE_ENFORCE)

    n = 0
    a_blocked = a_modified = 0
    b_blocked = b_modified = 0
    agree = a_only = b_only = both = neither = 0

    for i, action in enumerate(_iter_actions(actions_path)):
        ctx = SafetyContext(mode="profile-compare", step=i)
        da = safe_evaluate(sup_a, action, context=ctx).decision
        db = safe_evaluate(sup_b, action, context=ctx).decision
        n += 1
        if not da.allowed:
            a_blocked += 1
        if not db.allowed:
            b_blocked += 1
        if da.action_modified:
            a_modified += 1
        if db.action_modified:
            b_modified += 1
        if da.allowed == db.allowed:
            agree += 1
        if not da.allowed and db.allowed:
            a_only += 1
        if da.allowed and not db.allowed:
            b_only += 1
        if not da.allowed and not db.allowed:
            both += 1
        if da.allowed and db.allowed:
            neither += 1

    agreement_rate = (agree / n) if n else 1.0
    return {
        "schema_version": COMPARISON_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "profile_a": profile_a.name,
        "profile_b": profile_b.name,
        "actions_supervised": n,
        "agreement_rate": round(agreement_rate, 6),
        "a": {"blocked": a_blocked, "modified": a_modified, "passed": a_blocked == 0},
        "b": {"blocked": b_blocked, "modified": b_modified, "passed": b_blocked == 0},
        "breakdown": {
            "a_only_blocks": a_only, "b_only_blocks": b_only,
            "both_block": both, "neither_block": neither,
        },
        "claims": {
            "proves_profile_equivalence": False,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
        },
    }


def build_comparison_markdown(report: dict[str, Any]) -> str:
    a, b = report.get("a", {}), report.get("b", {})
    bd = report.get("breakdown", {})
    return (
        "# Profile Comparison Report\n\n"
        f"- Profile A: {report.get('profile_a')}\n"
        f"- Profile B: {report.get('profile_b')}\n"
        f"- Actions supervised: {report.get('actions_supervised')}\n"
        f"- Agreement rate: {report.get('agreement_rate')}\n\n"
        "## Per-profile\n\n"
        f"- A blocked: {a.get('blocked')} (passed={a.get('passed')})\n"
        f"- B blocked: {b.get('blocked')} (passed={b.get('passed')})\n\n"
        "## Breakdown\n\n"
        f"- A-only blocks: {bd.get('a_only_blocks')}\n"
        f"- B-only blocks: {bd.get('b_only_blocks')}\n"
        f"- Both block: {bd.get('both_block')}\n"
        f"- Neither block: {bd.get('neither_block')}\n\n"
        "## Claims\n\n"
        "This compares two software profiles on the provided actions. "
        "It does not prove profile equivalence or physical safety.\n"
    )


def build_profile_fit(summary: dict[str, Any]) -> dict[str, Any]:
    """Build a profile_fit.json from a safety summary (supervisor-check)."""
    n = summary.get("actions_supervised", 0) or 0
    def _rate(key):
        return round((summary.get(key, 0) or 0) / n, 6) if n else 0.0
    allowed_rate = _rate("actions_allowed")
    blocked_rate = _rate("actions_blocked")
    modified_rate = _rate("actions_modified")
    would_block_rate = _rate("would_block_actions")
    return {
        "profile": summary.get("profile"),
        "actions_supervised": n,
        "fit": {
            "allowed_rate": allowed_rate,
            "blocked_rate": blocked_rate,
            "modified_rate": modified_rate,
            "would_block_rate": would_block_rate,
        },
        "recommendation": {
            "profile_is_too_strict": blocked_rate > 0.05,
            "profile_is_too_loose": None,
        },
        "claims": {
            "proves_profile_fit_to_observed_actions": True,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
        },
    }


def build_profile_fit_markdown(fit: dict[str, Any]) -> str:
    f = fit.get("fit", {})
    return (
        "# Profile Fit\n\n"
        f"- Profile: {fit.get('profile')}\n"
        f"- Actions supervised: {fit.get('actions_supervised')}\n"
        f"- Allowed rate: {f.get('allowed_rate')}\n"
        f"- Blocked rate: {f.get('blocked_rate')}\n"
        f"- Modified rate: {f.get('modified_rate')}\n"
        f"- Would-block rate: {f.get('would_block_rate')}\n\n"
        "This is a software profile fit report. It does not prove physical safety.\n"
    )
