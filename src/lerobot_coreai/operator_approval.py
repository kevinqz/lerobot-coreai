# operator_approval.py — operator approval protocol for evidence bundles (v0.9.3).
#
# "Approve the evidence, not just the code." A named operator explicitly approves
# a specific simulator-only evidence bundle, bound to artifact SHA256 hashes,
# with a scope and an expiry. This is SOFTWARE evidence review only: it does not
# prove physical robot safety and does not authorize real-world actuation.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError

OPERATOR_APPROVAL_SCHEMA_VERSION = "lerobot-coreai.operator_approval.v0"

APPROVAL_SCOPES = (
    "sim_only",
    "sim_to_guarded_real_readiness",
    "guarded_real_dry_run",
    "guarded_real_single_session",
)

ATTESTATION_TEXT = (
    "I understand this approval is for software evidence readiness only and does "
    "not prove physical robot safety or authorize unrestricted real-world actuation."
)

# Artifacts under source_run/ considered for approval.
_REQUIRED_ARTIFACTS = {
    "sim_report": "sim_report.json",
    "safety_summary": "safety_summary.json",
    "safety_quality_report": "safety_quality_report.json",
}
_REGRESSION_ARTIFACT = ("safety_regression_report", "safety_regression_report.json")
_PROFILE_ARTIFACTS = {  # at least one required
    "calibrated_profile": "calibrated_profile.json",
    "profile_calibration_report": "profile_calibration_report.json",
    "profile_fit": "profile_fit.json",
}
_OPTIONAL_ARTIFACTS = {
    "safety_summary_md": "safety_summary.md",
    "profile_comparison_report": "profile_comparison_report.json",
    "failure_taxonomy": "failure_taxonomy.json",
}


@dataclass
class ApprovalConfig:
    bundle_dir: Path
    output_dir: Path | None = None
    operator: str | None = None
    approval_scope: str = "sim_to_guarded_real_readiness"
    expires_days: int | None = 30
    expires_at: str | None = None
    notes: str | None = None
    require_safety_quality_passed: bool = True
    require_safety_regression_passed: bool = True
    require_bundle_verified: bool = True
    allow_missing_regression: bool = False
    allow_missing_calibration: bool = False
    allow_warnings: bool = False
    attest_not_physical_safety: bool = False
    attest_not_unrestricted_actuation: bool = False


@dataclass
class ApprovalCheck:
    name: str
    passed: bool
    severity: str = "required"
    message: str = ""
    artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "severity": self.severity,
                "message": self.message, "artifact": self.artifact}


@dataclass
class ApprovalRequest:
    ok: bool
    checks: list[ApprovalCheck]
    artifacts: dict[str, dict[str, str]]
    warnings: list[str]
    approval_manifest_draft: dict[str, Any]
    # v0.9.4: distinguish "required checks passed" from "warnings present" so the
    # UX is unambiguous (ok = required passed AND (warnings allowed or none)).
    required_checks_passed: bool = True
    warnings_present: bool = False


@dataclass
class ApprovalVerificationResult:
    ok: bool
    approval_valid: bool
    expired: bool
    checksum_matches: bool
    checks: list[ApprovalCheck]
    warnings: list[str] = field(default_factory=list)


# MARK: - hashing / artifact collection

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def collect_approval_artifacts(bundle_dir: Path) -> dict[str, dict[str, str]]:
    """Collect present approval artifacts under source_run/ with their hashes."""
    bundle_dir = Path(bundle_dir)
    src = bundle_dir / "source_run"
    out: dict[str, dict[str, str]] = {}
    all_names = {**_REQUIRED_ARTIFACTS, _REGRESSION_ARTIFACT[0]: _REGRESSION_ARTIFACT[1],
                 **_PROFILE_ARTIFACTS, **_OPTIONAL_ARTIFACTS}
    for key, fname in all_names.items():
        p = src / fname
        if p.is_file():
            out[key] = {"path": f"source_run/{fname}", "sha256": sha256_file(p)}
    return out


def _overclaims(obj: dict[str, Any]) -> list[str]:
    """Return keys where an artifact overclaims physical/real-world safety."""
    claims = obj.get("claims", {}) or {}
    bad = []
    for key in ("proves_physical_safety", "proves_real_world_safety",
                "proves_real_task_success",
                "authorizes_unrestricted_real_world_actuation"):
        if key in claims and claims.get(key) is not False:
            bad.append(key)
    return bad


# MARK: - approval checks

def _run_approval_checks(
    bundle_dir: Path, config: ApprovalConfig,
) -> tuple[list[ApprovalCheck], dict[str, dict[str, str]], list[str]]:
    from .sim_bundle import verify_sim_bundle

    bundle_dir = Path(bundle_dir)
    src = bundle_dir / "source_run"
    checks: list[ApprovalCheck] = []
    warnings: list[str] = []

    def _check(name, passed, message="", severity="required", artifact=None):
        checks.append(ApprovalCheck(name=name, passed=passed, severity=severity,
                                    message=message, artifact=artifact))

    # Bundle presence + verification.
    manifest_path = bundle_dir / "bundle_manifest.json"
    checksums_path = bundle_dir / "checksums.json"
    _check("bundle_manifest_exists", manifest_path.is_file())
    _check("checksums_exists", checksums_path.is_file())
    if config.require_bundle_verified:
        try:
            v = verify_sim_bundle(bundle_dir)
            _check("bundle_verified", v.ok,
                   "" if v.ok else "; ".join(v.invariant_failures + v.checksum_failures))
        except Exception as e:
            _check("bundle_verified", False, str(e))

    # sim_report presence + invariants.
    sim_report_path = src / "sim_report.json"
    if sim_report_path.is_file():
        _check("sim_report_exists", True, artifact="source_run/sim_report.json")
        try:
            rep = _read_json(sim_report_path)
            _check("sim_report_mode_sim", rep.get("mode") == "sim")
            safety = rep.get("safety", {}) or {}
            _check("no_robot_egress", safety.get("robot_egress_enabled") is False)
            _check("actions_sent_to_robot_zero", safety.get("actions_sent_to_robot") == 0)
        except json.JSONDecodeError as e:
            _check("sim_report_mode_sim", False, f"unreadable: {e}")
    else:
        _check("sim_report_exists", False, artifact="source_run/sim_report.json")

    # safety_summary present.
    _check("safety_summary_exists", (src / "safety_summary.json").is_file(),
           artifact="source_run/safety_summary.json")

    # safety_quality report present + passed.
    sq_path = src / "safety_quality_report.json"
    _check("safety_quality_report_exists", sq_path.is_file(),
           artifact="source_run/safety_quality_report.json")
    if sq_path.is_file() and config.require_safety_quality_passed:
        try:
            sq = _read_json(sq_path)
            _check("safety_quality_passed", sq.get("passed") is True)
        except json.JSONDecodeError as e:
            _check("safety_quality_passed", False, f"unreadable: {e}")

    # safety_regression report present + passed (unless overridden).
    sr_path = src / _REGRESSION_ARTIFACT[1]
    if sr_path.is_file():
        _check("safety_regression_report_exists", True,
               artifact=f"source_run/{_REGRESSION_ARTIFACT[1]}")
        if config.require_safety_regression_passed:
            try:
                sr = _read_json(sr_path)
                _check("safety_regression_passed", sr.get("passed") is True)
            except json.JSONDecodeError as e:
                _check("safety_regression_passed", False, f"unreadable: {e}")
    elif config.allow_missing_regression:
        warnings.append("safety regression report missing but allowed by operator override")
        _check("safety_regression_report_exists", True, severity="waived",
               message="missing but allowed by --allow-missing-regression")
    else:
        _check("safety_regression_report_exists", False,
               artifact=f"source_run/{_REGRESSION_ARTIFACT[1]}")

    # A safety profile / calibration artifact present.
    has_profile = any((src / f).is_file() for f in _PROFILE_ARTIFACTS.values())
    if has_profile or config.allow_missing_calibration:
        if not has_profile:
            warnings.append("no profile/calibration artifact but allowed by override")
        _check("safety_profile_or_calibration_present", True,
               severity="required" if has_profile else "waived")
    else:
        _check("safety_profile_or_calibration_present", False)

    artifacts = collect_approval_artifacts(bundle_dir)

    # No overclaim in ANY collected JSON artifact (safety, profile, calibration,
    # comparison, fit, …) — not just the core safety reports. An unreadable
    # collected JSON is itself a failure (fail-closed).
    overclaim_found = []
    for key, ref in artifacts.items():
        rel = ref.get("path", "")
        if not rel.endswith(".json"):
            continue
        p = bundle_dir / rel
        if not p.is_file():
            continue
        try:
            bad = _overclaims(_read_json(p))
            if bad:
                overclaim_found.append(f"{rel}: {', '.join(bad)}")
        except json.JSONDecodeError:
            overclaim_found.append(f"{rel}: unreadable json")
    _check("no_physical_safety_overclaim", not overclaim_found,
           "; ".join(overclaim_found))

    return checks, artifacts, warnings


def _required_failed(checks: list[ApprovalCheck]) -> list[ApprovalCheck]:
    return [c for c in checks if c.severity == "required" and not c.passed]


# MARK: - build request / approve

def build_approval_request(config: ApprovalConfig) -> ApprovalRequest:
    """Run the approval checklist without approving. Produces a draft manifest."""
    bundle_dir = Path(config.bundle_dir)
    if not (bundle_dir / "bundle_manifest.json").is_file():
        raise CoreAIPolicyError(
            f"Bundle manifest not found in {bundle_dir}. Package the run first.")
    checks, artifacts, warnings = _run_approval_checks(bundle_dir, config)
    required_checks_passed = not _required_failed(checks)
    warnings_present = bool(warnings)
    ok = required_checks_passed and (config.allow_warnings or not warnings_present)
    draft = _build_manifest_dict(config, bundle_dir, checks, artifacts, warnings,
                                 approved=False, operator=None)
    return ApprovalRequest(
        ok=ok, checks=checks, artifacts=artifacts, warnings=warnings,
        approval_manifest_draft=draft, required_checks_passed=required_checks_passed,
        warnings_present=warnings_present)


def approve_bundle(config: ApprovalConfig) -> dict[str, Any]:
    """Create an approval manifest (approved=true) if all required checks pass.

    Requires explicit operator attestation. Raises CoreAIPolicyError otherwise.
    """
    bundle_dir = Path(config.bundle_dir)
    if not config.operator:
        raise CoreAIPolicyError("Approval requires --operator.")
    if not (config.attest_not_physical_safety and config.attest_not_unrestricted_actuation):
        raise CoreAIPolicyError(
            "Approval requires explicit attestation. Pass "
            "--i-understand-this-does-not-prove-physical-safety and "
            "--i-understand-this-does-not-authorize-unrestricted-real-world-actuation."
        )
    if config.approval_scope not in APPROVAL_SCOPES:
        raise CoreAIPolicyError(
            f"Unknown approval_scope: {config.approval_scope}. "
            f"Allowed: {', '.join(APPROVAL_SCOPES)}.")

    checks, artifacts, warnings = _run_approval_checks(bundle_dir, config)
    failed = _required_failed(checks)
    if failed:
        raise CoreAIPolicyError(
            "Cannot approve: required checks failed: "
            + "; ".join(c.name for c in failed))
    if warnings and not config.allow_warnings:
        raise CoreAIPolicyError(
            "Cannot approve: warnings present (use --allow-warnings to override): "
            + "; ".join(warnings))

    return _build_manifest_dict(config, bundle_dir, checks, artifacts, warnings,
                                approved=True, operator=config.operator)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_manifest_dict(
    config: ApprovalConfig, bundle_dir: Path, checks: list[ApprovalCheck],
    artifacts: dict[str, dict[str, str]], warnings: list[str], *,
    approved: bool, operator: str | None,
) -> dict[str, Any]:
    manifest_hash = sha256_file(bundle_dir / "bundle_manifest.json")
    checksums_hash = (sha256_file(bundle_dir / "checksums.json")
                      if (bundle_dir / "checksums.json").is_file() else None)
    created = _now()
    if config.expires_at:
        expires_at = config.expires_at
    elif config.expires_days is not None:
        expires_at = (created + timedelta(days=config.expires_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    else:
        expires_at = (created + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    approval_id = f"approval_{created.strftime('%Y%m%d')}_{manifest_hash.split(':')[1][:8]}"

    return {
        "schema_version": OPERATOR_APPROVAL_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "approval_id": approval_id,
        "created_at": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "approved": approved,
        "approved_by": operator,
        "approval_scope": config.approval_scope,
        "expires_at": expires_at,
        "bundle": {
            "path": str(bundle_dir),
            "manifest_sha256": manifest_hash,
            "checksums_sha256": checksums_hash,
        },
        "artifacts": artifacts,
        "checks": [c.to_dict() for c in checks],
        "operator_overrides": {
            "allow_missing_regression": config.allow_missing_regression,
            "allow_missing_calibration": config.allow_missing_calibration,
            "allow_warnings": config.allow_warnings,
        },
        "operator_attestation": {
            "text": ATTESTATION_TEXT,
            "accepted": bool(approved),
        },
        "claims": {
            "proves_operator_reviewed_evidence": approved,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
        "warnings": warnings,
        "notes": config.notes,
    }


def build_approval_checklist_markdown(request: ApprovalRequest) -> str:
    lines = ["# Operator Approval Checklist\n"]
    lines.append(f"Ready for approval: {request.ok}\n")
    lines.append("## Required checks\n")
    for c in request.checks:
        mark = "✓" if c.passed else "✗"
        lines.append(f"- {mark} {c.name}{(' — ' + c.message) if c.message else ''}")
    lines.append("\n## Warnings\n")
    lines.extend(f"- {w}" for w in (request.warnings or ["None"]))
    lines.append(
        "\n## What approval means\n\n"
        "Approval means a named operator reviewed this software evidence bundle "
        "and accepted it for the declared scope. It does not prove physical robot "
        "safety and does not authorize unrestricted real-world actuation.\n")
    return "\n".join(lines) + "\n"


# MARK: - verification

def _load_schema(name: str) -> dict[str, Any]:
    from importlib.resources import files
    return json.loads(files("lerobot_coreai.schemas").joinpath(name).read_text())


def verify_approval(bundle_dir: Path, approval_path: Path) -> ApprovalVerificationResult:
    """Verify an approval manifest against a bundle: schema, expiry, hashes, claims."""
    bundle_dir = Path(bundle_dir)
    approval_path = Path(approval_path)
    checks: list[ApprovalCheck] = []
    warnings: list[str] = []

    def _check(name, passed, message="", artifact=None):
        checks.append(ApprovalCheck(name=name, passed=passed, message=message, artifact=artifact))

    if not approval_path.is_file():
        return ApprovalVerificationResult(
            ok=False, approval_valid=False, expired=False, checksum_matches=False,
            checks=[ApprovalCheck("approval_exists", False, message="approval file not found")])
    approval = _read_json(approval_path)

    # Schema.
    try:
        import jsonschema
        jsonschema.validate(approval, _load_schema("operator-approval.schema.json"))
        _check("approval_schema_valid", True)
    except Exception as e:
        _check("approval_schema_valid", False, getattr(e, "message", str(e)))

    _check("approved", approval.get("approved") is True)
    att = approval.get("operator_attestation", {}) or {}
    _check("operator_attestation_accepted", att.get("accepted") is True)
    _check("approval_scope_valid", approval.get("approval_scope") in APPROVAL_SCOPES)

    # Expiry.
    expired = False
    expires_at = approval.get("expires_at")
    try:
        exp = datetime.strptime(expires_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        expired = _now() > exp
    except (TypeError, ValueError):
        expired = True
    _check("approval_not_expired", not expired,
           "" if not expired else f"expired at {expires_at}")

    # Hash binding.
    manifest_path = bundle_dir / "bundle_manifest.json"
    checksum_matches = True
    if manifest_path.is_file():
        actual = sha256_file(manifest_path)
        ok = actual == approval.get("bundle", {}).get("manifest_sha256")
        checksum_matches = checksum_matches and ok
        _check("bundle_manifest_hash_matches", ok)
    else:
        checksum_matches = False
        _check("bundle_manifest_hash_matches", False, "bundle_manifest.json missing")

    cs_path = bundle_dir / "checksums.json"
    approval_cs = approval.get("bundle", {}).get("checksums_sha256")
    if approval_cs is not None:
        ok = cs_path.is_file() and sha256_file(cs_path) == approval_cs
        checksum_matches = checksum_matches and ok
        _check("checksums_hash_matches", ok)

    # Artifact hashes.
    for key, ref in (approval.get("artifacts", {}) or {}).items():
        p = bundle_dir / ref.get("path", "")
        if not p.is_file():
            checksum_matches = False
            _check(f"artifact_present:{key}", False, "missing", artifact=ref.get("path"))
            continue
        ok = sha256_file(p) == ref.get("sha256")
        checksum_matches = checksum_matches and ok
        _check(f"artifact_hash:{key}", ok,
               "" if ok else "hash mismatch", artifact=ref.get("path"))

    # No overclaim in the approval itself.
    _check("no_physical_safety_overclaim", not _overclaims(approval))

    # Bundle still verifies.
    try:
        from .sim_bundle import verify_sim_bundle
        v = verify_sim_bundle(bundle_dir)
        _check("bundle_still_verifies", v.ok)
    except Exception as e:
        _check("bundle_still_verifies", False, str(e))

    # Re-run the required approval checks against the CURRENT bundle, so a forged
    # or minimal manifest can't pass just because its listed hashes match. The
    # approval's own overrides are honored (a waiver recorded at approval time).
    overrides = approval.get("operator_overrides", {}) or {}
    try:
        cfg = ApprovalConfig(
            bundle_dir=bundle_dir,
            allow_missing_regression=overrides.get("allow_missing_regression", False),
            allow_missing_calibration=overrides.get("allow_missing_calibration", False),
            allow_warnings=overrides.get("allow_warnings", False),
        )
        rechecks, _, _ = _run_approval_checks(bundle_dir, cfg)
        failed = _required_failed(rechecks)
        _check("approval_required_checks_still_pass", not failed,
               "; ".join(c.name for c in failed))
    except Exception as e:
        _check("approval_required_checks_still_pass", False, str(e))

    # The approval must actually BIND the required artifacts (not just claim
    # approved with an empty/minimal artifact set).
    required_keys = {"sim_report", "safety_summary", "safety_quality_report"}
    if not overrides.get("allow_missing_regression"):
        required_keys.add("safety_regression_report")
    bound = set((approval.get("artifacts", {}) or {}).keys())
    missing_bound = sorted(required_keys - bound)
    _check("required_artifacts_bound", not missing_bound,
           f"approval does not bind required artifacts: {', '.join(missing_bound)}"
           if missing_bound else "")

    approval_valid = all(c.passed for c in checks)
    return ApprovalVerificationResult(
        ok=approval_valid, approval_valid=approval_valid, expired=expired,
        checksum_matches=checksum_matches, checks=checks, warnings=warnings)
