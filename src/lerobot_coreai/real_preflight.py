# real_preflight.py — guarded real-mode preflight (v1.0.0).
#
# Verifies that a guarded real session COULD start. It never sends an action.
# Every real egress is gated on these checks: no verified readiness, no valid
# approval, no safety profile, no supervisor enforce, no operator attestation,
# no bounded session, no adapter preflight -> no real egress.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError

REAL_PREFLIGHT_SCHEMA_VERSION = "lerobot-coreai.real_preflight.v0"

MAX_FPS = 10.0
MAX_STEPS_CAP = 100_000
MAX_DURATION_S = 3600.0

# A safety profile is only usable in real mode if it declares one of these
# intended modes. A sim/shadow-only profile must not gate real egress.
REAL_INTENDED_MODES = {
    "real", "guarded_real", "guarded_real_dry_run", "guarded_real_single_session",
}


@dataclass
class RealPreflightConfig:
    mode: str
    policy_path: str
    runner_url: str
    robot_adapter: str
    robot_type: str
    safety_profile: Path
    readiness_report: Path
    approval: Path
    bundle_dir: Path
    robot_config: Path | None = None
    robot_endpoint: str | None = None
    robot_token: str | None = None
    operator: str | None = None
    max_steps: int | None = None
    duration_seconds: float | None = None
    fps: float | None = None
    attest_real_hardware: bool = False
    attest_physical_estop: bool = False
    attest_workspace_clear: bool = False


@dataclass
class RealPreflightCheck:
    name: str
    passed: bool
    severity: str = "required"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed,
                "severity": self.severity, "message": self.message}


@dataclass
class RealPreflightResult:
    ok: bool
    checks: list[RealPreflightCheck] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _path_eq(a: str | Path, b: str | Path) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def evaluate_real_preflight(config: RealPreflightConfig) -> RealPreflightResult:
    """Run all real-mode gates. Never sends an action."""
    from .operator_approval import verify_approval
    from .safety_profiles import resolve_safety_profile
    from .sim_bundle import verify_sim_bundle
    from .robot_adapters import build_robot_adapter

    checks: list[RealPreflightCheck] = []
    guarded = config.mode == "guarded"

    def _c(name, passed, message="", severity="required"):
        checks.append(RealPreflightCheck(name=name, passed=passed, message=message,
                                         severity=severity))

    _c("mode_valid", config.mode in ("preflight", "guarded"))

    # Readiness report — must be a valid, coherent report for THIS bundle/approval.
    rr_path = Path(config.readiness_report)
    if not rr_path.is_file():
        _c("readiness_report_exists", False, str(rr_path))
    else:
        _c("readiness_report_exists", True)
        try:
            rr = _read_json(rr_path)
        except json.JSONDecodeError as e:
            _c("readiness_schema_valid", False, f"unreadable: {e}")
            rr = None
        if rr is not None:
            # Schema.
            try:
                import jsonschema
                from importlib.resources import files
                schema = json.loads(files("lerobot_coreai.schemas").joinpath(
                    "release-readiness-report.schema.json").read_text())
                jsonschema.validate(rr, schema)
                _c("readiness_schema_valid", True)
            except Exception as e:
                _c("readiness_schema_valid", False, getattr(e, "message", str(e)))
            _c("readiness_report_ready_true", rr.get("ready") is True)
            _c("readiness_bundle_verified",
               (rr.get("bundle", {}) or {}).get("verified") is True)
            _c("readiness_approval_valid",
               (rr.get("approval", {}) or {}).get("valid") is True)
            ev = rr.get("evidence", {}) or {}
            _c("readiness_evidence_safety_quality_passed",
               ev.get("safety_quality_passed") is True)
            _c("readiness_evidence_safety_regression_passed",
               ev.get("safety_regression_passed") is True)
            claims = rr.get("claims", {}) or {}
            _c("readiness_no_overclaim",
               claims.get("proves_physical_safety") is False
               and claims.get("proves_real_world_safety") is False
               and claims.get("authorizes_unrestricted_real_world_actuation") is False)
            # Path coherence: the readiness report must reference THIS bundle/approval.
            rb = (rr.get("bundle", {}) or {}).get("path")
            ra = (rr.get("approval", {}) or {}).get("path")
            bundle_match = rb is None or _path_eq(rb, config.bundle_dir)
            approval_match = ra is None or _path_eq(ra, config.approval)
            _c("readiness_bundle_path_matches", bundle_match,
               "" if bundle_match else f"{rb} != {config.bundle_dir}")
            _c("readiness_approval_path_matches", approval_match,
               "" if approval_match else f"{ra} != {config.approval}")

    # Bundle.
    bundle_dir = Path(config.bundle_dir)
    _c("bundle_dir_exists", bundle_dir.is_dir())
    bundle_ok = False
    if bundle_dir.is_dir():
        try:
            bundle_ok = verify_sim_bundle(bundle_dir).ok
        except Exception as e:
            _c("bundle_verifies", False, str(e))
    if bundle_dir.is_dir():
        _c("bundle_verifies", bundle_ok)

    # Approval (bound to the bundle).
    ap_path = Path(config.approval)
    if not ap_path.is_file():
        _c("approval_exists", False, str(ap_path))
    else:
        _c("approval_exists", True)
        try:
            av = verify_approval(bundle_dir, ap_path)
            _c("approval_valid", av.approval_valid)
            _c("approval_not_expired", not av.expired)
        except Exception as e:
            _c("approval_valid", False, str(e))

    _c("policy_path_present", bool(config.policy_path))
    _c("runner_url_present", bool(config.runner_url))

    # Safety profile.
    profile_action_shape: list[int] | None = None
    sp_path = Path(config.safety_profile) if config.safety_profile else None
    if not sp_path or not sp_path.is_file():
        _c("safety_profile_exists", False, str(sp_path))
    else:
        _c("safety_profile_exists", True)
        try:
            profile = resolve_safety_profile(path=sp_path)
            profile_action_shape = profile.action_shape
            _c("safety_profile_valid", True)
            rt_ok = profile.robot_type is None or profile.robot_type == config.robot_type
            _c("safety_profile_robot_type_matches", rt_ok,
               "" if rt_ok else f"profile {profile.robot_type} != {config.robot_type}")
            # A sim/shadow-only profile must not gate real egress.
            intended = set(profile.intended_modes or [])
            real_ok = bool(intended & REAL_INTENDED_MODES)
            _c("safety_profile_intended_for_real", real_ok,
               "" if real_ok else f"intended_modes={sorted(intended)} lacks a guarded-real mode")
        except CoreAIPolicyError as e:
            _c("safety_profile_valid", False, str(e))

    # For a non-mock (real) adapter in guarded mode, do NOT touch the external
    # controller's /preflight endpoint until the basic guarded attestations are
    # present — an operator must attest before we contact real hardware at all.
    guarded_attest_ok = (
        bool(config.operator) and config.attest_real_hardware
        and config.attest_physical_estop and config.attest_workspace_clear
        and config.max_steps is not None and config.fps is not None)

    # Robot adapter (preflight only — no connect/send).
    try:
        adapter = build_robot_adapter(
            config.robot_adapter, config.robot_type,
            endpoint=config.robot_endpoint, config=config.robot_config,
            token=config.robot_token)
        _c("robot_adapter_known", True)
        if guarded and config.robot_adapter != "mock" and not guarded_attest_ok:
            _c("robot_adapter_preflight_passes", False,
               "skipped: guarded attestations missing before contacting real adapter")
        else:
            pf = adapter.preflight()
            _c("robot_adapter_preflight_passes", bool(pf.get("ok")),
               "" if pf.get("ok") else str(pf))
            # v1.0.3: an external-http controller must satisfy the capability
            # contract, coherent with robot type / profile shape / fps.
            if config.robot_adapter == "external-http" and pf.get("ok"):
                from .external_http_contract import validate_controller_preflight
                for name, ok, msg in validate_controller_preflight(
                        pf, robot_type=config.robot_type,
                        profile_action_shape=profile_action_shape,
                        requested_fps=config.fps):
                    _c(name, ok, msg)
    except CoreAIPolicyError as e:
        _c("robot_adapter_known", False, str(e))

    # Supervisor is always enforce in real mode — asserted here for the record.
    _c("supervisor_enforce", True, "hardcoded enforce")

    # Guarded-only requirements.
    if guarded:
        _c("operator_present", bool(config.operator))
        _c("attestation_real_hardware", config.attest_real_hardware)
        _c("attestation_physical_estop", config.attest_physical_estop)
        _c("attestation_workspace_clear", config.attest_workspace_clear)
        _c("max_steps_present", config.max_steps is not None)
        if config.max_steps is not None:
            _c("max_steps_bounded", 0 < config.max_steps <= MAX_STEPS_CAP,
               f"1..{MAX_STEPS_CAP}")
        fps = config.fps if config.fps is not None else 0
        _c("fps_bounded", 0 < fps <= MAX_FPS, f"0 < fps <= {MAX_FPS}")
        if config.duration_seconds is not None:
            _c("duration_bounded", 0 < config.duration_seconds <= MAX_DURATION_S,
               f"0 < duration <= {MAX_DURATION_S}")

    ok = all(c.passed for c in checks if c.severity == "required")
    report = {
        "schema_version": REAL_PREFLIGHT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "mode": config.mode,
        "actions_sent_to_robot": 0,
        "checks": [c.to_dict() for c in checks],
        "claims": {
            "proves_preflight_passed": ok,
            "proves_physical_safety": False,
            "proves_real_world_safety": False,
            "authorizes_unrestricted_real_world_actuation": False,
        },
    }
    return RealPreflightResult(ok=ok, checks=checks, report=report)
