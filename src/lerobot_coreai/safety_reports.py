# safety_reports.py — safety supervisor reports (v0.9.0).
#
# Produces the per-action safety_report.jsonl, the aggregate safety_summary.json,
# and a human safety_summary.md. These are SOFTWARE supervision records — they do
# not prove physical robot safety or real-world task success.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .reports import now_iso
from .safety_supervisor import SafetyContext, SafetyDecision

SAFETY_SUMMARY_SCHEMA_VERSION = "lerobot-coreai.safety_summary.v0"


def decision_record(
    decision: SafetyDecision, *, context: SafetyContext | None = None,
) -> dict[str, Any]:
    """Build one safety_report.jsonl line from a decision (+ context)."""
    rec: dict[str, Any] = {
        "timestamp": now_iso(),
        "mode": context.mode if context else decision.mode,
        "episode": context.episode if context else None,
        "step": context.step if context else None,
        "allowed": decision.allowed,
        "action_modified": decision.action_modified,
        "severity": decision.severity,
        "reasons": decision.reasons,
        "checks": decision.checks,
        "profile": decision.profile,
        "original_action_shape": decision.original_action_shape,
        "supervised_action_shape": decision.supervised_action_shape,
    }
    return rec


def append_safety_decision(
    path: Path, decision: SafetyDecision, *, context: SafetyContext | None = None,
) -> None:
    """Append a single decision record to safety_report.jsonl."""
    with open(path, "a") as f:
        f.write(json.dumps(decision_record(decision, context=context)) + "\n")


@dataclass
class SafetyAccumulator:
    """Aggregates decisions across a run into summary counts."""

    profile: str
    mode: str
    actions_supervised: int = 0
    actions_allowed: int = 0
    actions_blocked: int = 0
    actions_modified: int = 0
    critical_failures: int = 0
    # Critical findings that did NOT block egress (report_only). These must not
    # be masked: a report_only run that found an unsafe action does not pass.
    would_block_actions: int = 0
    critical_findings: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, decision: SafetyDecision) -> None:
        self.actions_supervised += 1
        if decision.allowed:
            self.actions_allowed += 1
        else:
            self.actions_blocked += 1
        if decision.action_modified:
            self.actions_modified += 1
        # A critical finding is any critical-severity decision, whether or not
        # it was operationally blocked (report_only allows but still flags it).
        if decision.severity == "critical":
            self.critical_findings += 1
        if decision.severity == "critical" and not decision.allowed:
            self.critical_failures += 1
        if "report_only_would_block" in decision.reasons:
            self.would_block_actions += 1
        for r in decision.reasons:
            # Count the meaningful failure/modification reasons.
            self.reasons[r] = self.reasons.get(r, 0) + 1

    def top_reasons(self, limit: int = 10) -> dict[str, int]:
        return dict(sorted(self.reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:limit])

    @property
    def passed(self) -> bool:
        # Fail if anything was blocked OR any critical finding was seen — even in
        # report_only, where nothing is blocked but findings must surface.
        return (
            self.actions_blocked == 0
            and self.critical_failures == 0
            and self.critical_findings == 0
            and self.would_block_actions == 0
        )


def build_safety_summary(acc: SafetyAccumulator) -> dict[str, Any]:
    """Build the safety_summary.json dict (schema safety_summary.v0)."""
    return {
        "schema_version": SAFETY_SUMMARY_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "profile": acc.profile,
        "mode": acc.mode,
        "actions_supervised": acc.actions_supervised,
        "actions_allowed": acc.actions_allowed,
        "actions_blocked": acc.actions_blocked,
        "actions_modified": acc.actions_modified,
        "critical_failures": acc.critical_failures,
        "would_block_actions": acc.would_block_actions,
        "critical_findings": acc.critical_findings,
        "top_reasons": acc.top_reasons(),
        "passed": acc.passed,
        "claims": {
            "proves_software_supervision": True,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "proves_real_task_success": False,
        },
    }


def build_safety_summary_markdown(summary: dict[str, Any]) -> str:
    """Build the human safety_summary.md."""
    top = summary.get("top_reasons", {}) or {}
    reason_lines = "\n".join(f"- {k}: {v}" for k, v in top.items()) or "- (none)"
    return (
        "# Safety Supervisor Summary\n\n"
        f"- Profile: {summary.get('profile')}\n"
        f"- Mode: {summary.get('mode')}\n"
        f"- Actions supervised: {summary.get('actions_supervised')}\n"
        f"- Allowed: {summary.get('actions_allowed')}\n"
        f"- Blocked: {summary.get('actions_blocked')}\n"
        f"- Would block (report_only): {summary.get('would_block_actions', 0)}\n"
        f"- Modified: {summary.get('actions_modified')}\n"
        f"- Critical failures: {summary.get('critical_failures')}\n"
        f"- Critical findings: {summary.get('critical_findings', 0)}\n"
        f"- Passed: {summary.get('passed')}\n\n"
        "## Top reasons\n\n"
        f"{reason_lines}\n\n"
        "## Claims\n\n"
        "This is a software runtime supervision report. "
        "It does not prove physical robot safety. "
        "It does not prove real-world task success.\n"
    )
