# safety_quality.py — supervisor safety quality gates (v0.9.2).
#
# Turns a safety_summary.json (from the runtime supervisor / supervisor-check)
# into a pass/fail CI decision. These are SOFTWARE gates: they can prove a
# specific artifact met configured thresholds, nothing more. They do not prove
# physical robot safety, real-world task success, or future policy safety.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError

SAFETY_QUALITY_REPORT_SCHEMA_VERSION = "lerobot-coreai.safety_quality_report.v0"


@dataclass
class SafetyQualityConfig:
    """Thresholds for safety quality gates. None disables a check."""

    max_actions_blocked: int | None = 0
    max_block_rate: float | None = 0.0
    max_critical_failures: int | None = 0
    max_critical_findings: int | None = 0
    max_would_block_actions: int | None = 0
    max_would_block_rate: float | None = 0.0
    max_actions_modified: int | None = None
    max_modification_rate: float | None = None
    max_clip_rate: float | None = None
    max_delta_failures: int | None = 0
    max_shape_failures: int | None = 0
    max_nonfinite_failures: int | None = 0
    require_passed_summary: bool = True
    require_zero_parse_errors: bool = True
    min_actions_supervised: int | None = None
    fail_on_safety_quality: bool = False


@dataclass
class SafetyQualityCheck:
    name: str
    passed: bool
    value: Any
    threshold: Any
    severity: str = "error"
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "passed": self.passed, "value": self.value,
            "threshold": self.threshold, "severity": self.severity,
            "message": self.message,
        }


@dataclass
class SafetyQualityResult:
    passed: bool
    checks: list[SafetyQualityCheck]
    summary: dict[str, Any]
    report: dict[str, Any] = field(default_factory=dict)


# MARK: - Input resolution

def load_safety_summary_from_path(
    *, safety_summary: Path | None = None, sim_report: Path | None = None,
    profile_fit: Path | None = None, run_dir: Path | None = None,
    bundle_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a safety summary dict from one of several inputs.

    Returns (summary, source_meta). Fail-closed on missing/unreadable input.
    """
    def _read(p: Path) -> dict[str, Any]:
        if not p.is_file():
            raise CoreAIPolicyError(f"Safety input not found: {p}")
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError as e:
            raise CoreAIPolicyError(f"Malformed safety input {p}: {e}") from None

    if safety_summary is not None:
        return _read(Path(safety_summary)), {"type": "safety_summary", "path": str(safety_summary)}
    if profile_fit is not None:
        # profile_fit carries fit rates but not raw counts; derive a summary.
        fit = _read(Path(profile_fit))
        return _summary_from_profile_fit(fit), {"type": "profile_fit", "path": str(profile_fit)}
    if sim_report is not None:
        rep = _read(Path(sim_report))
        return _summary_from_sim_report(rep), {"type": "sim_report", "path": str(sim_report)}
    if run_dir is not None:
        rd = Path(run_dir)
        for name in ("safety_summary.json", "sim_report.json"):
            p = rd / name
            if p.is_file():
                if name == "safety_summary.json":
                    return _read(p), {"type": "safety_summary", "path": str(p)}
                return _summary_from_sim_report(_read(p)), {"type": "sim_report", "path": str(p)}
        raise CoreAIPolicyError(f"No safety_summary.json or sim_report.json in {run_dir}")
    if bundle_dir is not None:
        src = Path(bundle_dir) / "source_run"
        for name in ("safety_summary.json", "sim_report.json"):
            p = src / name
            if p.is_file():
                if name == "safety_summary.json":
                    return _read(p), {"type": "safety_summary", "path": str(p)}
                return _summary_from_sim_report(_read(p)), {"type": "sim_report", "path": str(p)}
        raise CoreAIPolicyError(f"No safety summary in bundle {bundle_dir}/source_run")
    raise CoreAIPolicyError("No safety input provided (fail-closed).")


def _summary_from_sim_report(report: dict[str, Any]) -> dict[str, Any]:
    sec = report.get("safety_supervisor")
    if not sec:
        raise CoreAIPolicyError("sim_report has no safety_supervisor section.")
    return sec


def _summary_from_profile_fit(fit: dict[str, Any]) -> dict[str, Any]:
    n = fit.get("actions_supervised", 0) or 0
    f = fit.get("fit", {}) or {}
    def _count(rate_key):
        return round((f.get(rate_key, 0) or 0) * n)
    return {
        "actions_supervised": n,
        "actions_allowed": round((f.get("allowed_rate", 0) or 0) * n),
        "actions_blocked": _count("blocked_rate"),
        "actions_modified": _count("modified_rate"),
        "would_block_actions": _count("would_block_rate"),
        "critical_failures": _count("blocked_rate"),
        "critical_findings": _count("blocked_rate") + _count("would_block_rate"),
        "top_reasons": {},
        "passed": (f.get("blocked_rate", 0) or 0) == 0 and (f.get("would_block_rate", 0) or 0) == 0,
        "profile": fit.get("profile"),
    }


# MARK: - Evaluation

_REQUIRED_SUMMARY_FIELDS = (
    "actions_supervised", "actions_blocked", "actions_modified",
    "critical_failures", "critical_findings", "would_block_actions", "passed",
)


def evaluate_safety_quality(
    summary: dict[str, Any], config: SafetyQualityConfig,
) -> SafetyQualityResult:
    """Evaluate a safety summary against the configured gates.

    Fail-closed: a malformed summary (missing required fields) or a zero-action
    summary cannot pass a safety gate — those raise rather than silently
    defaulting counts to zero and passing.
    """
    missing = [k for k in _REQUIRED_SUMMARY_FIELDS if k not in summary]
    if missing:
        raise CoreAIPolicyError(
            f"Malformed safety summary: missing required fields: {', '.join(missing)}."
        )
    n_raw = summary.get("actions_supervised")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 1:
        raise CoreAIPolicyError(
            "Malformed safety summary: actions_supervised must be a positive integer "
            f"(got {n_raw!r})."
        )
    n = summary.get("actions_supervised", 0) or 0
    blocked = summary.get("actions_blocked", 0) or 0
    modified = summary.get("actions_modified", 0) or 0
    would_block = summary.get("would_block_actions", 0) or 0
    critical_failures = summary.get("critical_failures", 0) or 0
    critical_findings = summary.get("critical_findings", 0) or 0
    reasons = summary.get("top_reasons", {}) or {}

    def _rate(x):
        return (x / n) if n else 0.0

    block_rate = _rate(blocked)
    modification_rate = _rate(modified)
    would_block_rate = _rate(would_block)
    clip_count = reasons.get("action_clipped_to_bounds", 0)
    clip_rate = _rate(clip_count)
    delta_failures = reasons.get("delta", 0) + reasons.get("delta_unverifiable_shape_changed", 0)
    shape_failures = reasons.get("shape", 0)
    nonfinite_failures = reasons.get("finite", 0)
    parse_errors = reasons.get("unparseable_actions_line", 0)

    checks: list[SafetyQualityCheck] = []

    def _le(name, value, threshold):
        if threshold is None:
            return
        checks.append(SafetyQualityCheck(
            name=name, passed=value <= threshold, value=value, threshold=threshold,
            message=None if value <= threshold else f"{name} exceeded threshold",
        ))

    if config.require_passed_summary:
        sp = summary.get("passed")
        checks.append(SafetyQualityCheck(
            name="require_passed_summary", passed=sp is True, value=sp, threshold=True,
            message=None if sp is True else "safety summary did not pass",
        ))
    if config.min_actions_supervised is not None:
        ok = n >= config.min_actions_supervised
        checks.append(SafetyQualityCheck(
            name="min_actions_supervised", passed=ok, value=n,
            threshold=config.min_actions_supervised,
            message=None if ok else "too few actions supervised",
        ))
    _le("max_actions_blocked", blocked, config.max_actions_blocked)
    _le("max_block_rate", round(block_rate, 6), config.max_block_rate)
    _le("max_critical_failures", critical_failures, config.max_critical_failures)
    _le("max_critical_findings", critical_findings, config.max_critical_findings)
    _le("max_would_block_actions", would_block, config.max_would_block_actions)
    _le("max_would_block_rate", round(would_block_rate, 6), config.max_would_block_rate)
    _le("max_actions_modified", modified, config.max_actions_modified)
    _le("max_modification_rate", round(modification_rate, 6), config.max_modification_rate)
    _le("max_clip_rate", round(clip_rate, 6), config.max_clip_rate)
    _le("max_delta_failures", delta_failures, config.max_delta_failures)
    _le("max_shape_failures", shape_failures, config.max_shape_failures)
    _le("max_nonfinite_failures", nonfinite_failures, config.max_nonfinite_failures)
    if config.require_zero_parse_errors:
        ok = parse_errors == 0
        checks.append(SafetyQualityCheck(
            name="require_zero_parse_errors", passed=ok, value=parse_errors, threshold=0,
            message=None if ok else "unparseable action lines present",
        ))

    passed = all(c.passed for c in checks)
    derived = {
        "actions_supervised": n,
        "actions_blocked": blocked,
        "block_rate": round(block_rate, 6),
        "actions_modified": modified,
        "modification_rate": round(modification_rate, 6),
        "would_block_actions": would_block,
        "would_block_rate": round(would_block_rate, 6),
        "critical_failures": critical_failures,
        "critical_findings": critical_findings,
        "clip_rate": round(clip_rate, 6),
        "delta_failures": delta_failures,
        "shape_failures": shape_failures,
        "nonfinite_failures": nonfinite_failures,
        "parse_errors": parse_errors,
    }
    return SafetyQualityResult(passed=passed, checks=checks, summary=derived)


def build_safety_quality_report(
    result: SafetyQualityResult, *, source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SAFETY_QUALITY_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "passed": result.passed,
        "source": source or {},
        "summary": result.summary,
        "checks": [c.to_dict() for c in result.checks],
        "claims": {
            "proves_software_safety_quality": True,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "proves_real_task_success": False,
        },
    }


def build_safety_quality_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    failed = [c for c in report.get("checks", []) if not c["passed"]]
    fail_lines = "\n".join(
        f"- {c['name']}: {c['value']} > {c['threshold']}" for c in failed) or "- (none)"
    return (
        "# Safety Quality Report\n\n"
        f"Passed: {report.get('passed')}\n\n"
        "## Summary\n\n"
        f"- Actions supervised: {s.get('actions_supervised')}\n"
        f"- Blocked: {s.get('actions_blocked')}\n"
        f"- Block rate: {s.get('block_rate')}\n"
        f"- Modified: {s.get('actions_modified')}\n"
        f"- Modification rate: {s.get('modification_rate')}\n"
        f"- Critical findings: {s.get('critical_findings')}\n"
        f"- Would-block actions: {s.get('would_block_actions')}\n\n"
        "## Failed checks\n\n"
        f"{fail_lines}\n\n"
        "## Claims\n\n"
        "This is a software safety quality report. "
        "It does not prove physical robot safety. "
        "It does not prove real-world task success.\n"
    )
