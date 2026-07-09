# release_readiness.py — go/no-go release readiness evidence (v0.9.3).
#
# Combines bundle verification + operator approval + required evidence into a
# single readiness decision. Readiness is scoped to SOFTWARE evidence only. It
# does not prove physical robot safety and does not authorize real-world actuation.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .operator_approval import verify_approval

RELEASE_READINESS_REPORT_SCHEMA_VERSION = "lerobot-coreai.release_readiness_report.v0"


@dataclass
class ReleaseReadinessConfig:
    bundle_dir: Path
    approval_path: Path
    output_dir: Path | None = None
    # Release readiness is the go/no-go gate — stricter than approve-bundle.
    # By default a missing/failed safety regression BLOCKS readiness, even if the
    # operator waived it at approval time. Downgrade to a warning only when this
    # is explicitly set AND the approval itself carries the waiver.
    allow_missing_regression: bool = False


@dataclass
class ReleaseReadinessResult:
    ready: bool
    report: dict[str, Any]
    blocking_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def evaluate_release_readiness(
    bundle_dir: Path, approval_path: Path, *, allow_missing_regression: bool = False,
) -> ReleaseReadinessResult:
    """Produce a release-readiness decision from a bundle + approval."""
    bundle_dir = Path(bundle_dir)
    approval_path = Path(approval_path)
    src = bundle_dir / "source_run"
    blocking: list[str] = []
    warnings: list[str] = []

    # Bundle verification.
    from .sim_bundle import verify_sim_bundle
    try:
        bundle_v = verify_sim_bundle(bundle_dir)
        bundle_verified = bundle_v.ok
        if not bundle_verified:
            blocking.extend(f"bundle: {f}" for f in
                            (bundle_v.invariant_failures + bundle_v.checksum_failures))
    except Exception as e:
        bundle_verified = False
        blocking.append(f"bundle verification error: {e}")

    # Approval verification.
    approval_valid = False
    expired = False
    approved_by = None
    expires_at = None
    scope = None
    regression_waived_by_approval = False
    if not approval_path.is_file():
        blocking.append("approval manifest missing")
    else:
        approval = _read_json(approval_path)
        approved_by = approval.get("approved_by")
        expires_at = approval.get("expires_at")
        scope = approval.get("approval_scope")
        regression_waived_by_approval = (
            (approval.get("operator_overrides", {}) or {}).get("allow_missing_regression") is True
        )
        av = verify_approval(bundle_dir, approval_path)
        approval_valid = av.approval_valid
        expired = av.expired
        if not approval_valid:
            blocking.extend(f"approval: {c.name}" for c in av.checks if not c.passed)

    # Evidence presence + pass.
    def _passed(fname):
        p = src / fname
        if not p.is_file():
            return None
        try:
            return _read_json(p).get("passed")
        except json.JSONDecodeError:
            return None

    sq_passed = _passed("safety_quality_report.json")
    sr_passed = _passed("safety_regression_report.json")
    evidence = {
        "sim_report": (src / "sim_report.json").is_file(),
        "safety_summary": (src / "safety_summary.json").is_file(),
        "safety_quality_passed": sq_passed is True,
        "safety_regression_passed": sr_passed is True,
        "profile_calibration_present": (src / "profile_calibration_report.json").is_file()
        or (src / "calibrated_profile.json").is_file(),
        "bundle_checksums_valid": bundle_verified,
    }
    if not evidence["sim_report"]:
        blocking.append("evidence: sim_report missing")
    if not evidence["safety_summary"]:
        blocking.append("evidence: safety_summary missing")
    if sq_passed is not True:
        blocking.append("evidence: safety_quality not passed")
    if sr_passed is not True:
        # Release readiness is the go/no-go gate: block a missing/failed safety
        # regression by DEFAULT, even when the operator waived it at approval
        # time. Downgrade to a warning only when explicitly allowed here AND the
        # approval carries the waiver.
        msg = "evidence: safety_regression not passed/present"
        if allow_missing_regression and regression_waived_by_approval:
            warnings.append(msg + " (waived)")
        else:
            blocking.append(msg)

    ready = bundle_verified and approval_valid and not expired and not blocking
    report = build_release_readiness_report(
        bundle_dir=bundle_dir, approval_path=approval_path, ready=ready,
        scope=scope, approved_by=approved_by, expires_at=expires_at,
        approval_valid=approval_valid, bundle_verified=bundle_verified,
        evidence=evidence, blocking=blocking, warnings=warnings)
    return ReleaseReadinessResult(ready=ready, report=report,
                                  blocking_failures=blocking, warnings=warnings)


def build_release_readiness_report(
    *, bundle_dir: Path, approval_path: Path, ready: bool, scope: str | None,
    approved_by: str | None, expires_at: str | None, approval_valid: bool,
    bundle_verified: bool, evidence: dict[str, Any], blocking: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": RELEASE_READINESS_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ready": ready,
        "readiness_scope": scope or "sim_to_guarded_real_readiness",
        "bundle": {"path": str(bundle_dir), "verified": bundle_verified},
        "approval": {
            "path": str(approval_path),
            "valid": approval_valid,
            "approved_by": approved_by,
            "expires_at": expires_at,
        },
        "evidence": evidence,
        "blocking_failures": blocking,
        "warnings": warnings,
        "claims": {
            "proves_release_readiness_for_scope": ready,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
    }


def build_release_readiness_markdown(report: dict[str, Any]) -> str:
    ev = report.get("evidence", {})
    ap = report.get("approval", {})
    blocking = report.get("blocking_failures", [])
    block_lines = "\n".join(f"- {b}" for b in blocking) or "- None"
    return (
        "# Release Readiness Report\n\n"
        f"Ready: {report.get('ready')}\n"
        f"Scope: {report.get('readiness_scope')}\n\n"
        "## Evidence\n\n"
        f"- Bundle verified: {report.get('bundle', {}).get('verified')}\n"
        f"- Safety quality passed: {ev.get('safety_quality_passed')}\n"
        f"- Safety regression passed: {ev.get('safety_regression_passed')}\n"
        f"- Operator approval valid: {ap.get('valid')}\n"
        f"- Approved by: {ap.get('approved_by')}\n"
        f"- Approval expires: {ap.get('expires_at')}\n\n"
        "## Blocking failures\n\n"
        f"{block_lines}\n\n"
        "## Claims\n\n"
        "This report proves release-readiness only for the declared software "
        "evidence scope. It does not prove physical robot safety. It does not "
        "authorize unrestricted real-world actuation.\n"
    )
