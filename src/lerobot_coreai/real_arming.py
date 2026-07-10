# real_arming.py — arming manifest for guarded real sessions (v1.0.6).
#
# The arming manifest records exactly what was armed: which policy, which robot,
# which limits, which approval, and which readiness report (bound by SHA256).
# It is written the moment egress is armed — before the first action — so an
# operator/auditor can see the committed envelope independent of how the session
# ends. It proves nothing about physical safety or real-world success.

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

REAL_ARMING_SCHEMA_VERSION = "lerobot-coreai.real_arming.v0"


def _sha256_file(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_arming_manifest(*, session_id, created_at, operator, robot_adapter,
                          robot_type, policy_path, safety_profile, readiness_report,
                          approval, approval_id, max_steps, duration_seconds, fps,
                          deadman_timeout_s, deadman_enabled, attestations,
                          abort_file=None) -> dict[str, Any]:
    """Build the arming manifest. Paths are bound by SHA256 where present."""
    return {
        "schema_version": REAL_ARMING_SCHEMA_VERSION,
        "session_id": session_id,
        "armed_at": created_at,
        "operator": operator,
        "robot": {"adapter": robot_adapter, "type": robot_type},
        "policy_path": policy_path,
        "approval_id": approval_id,
        "limits": {
            "max_steps": max_steps,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "deadman_timeout_s": deadman_timeout_s,
            "deadman_enabled": deadman_enabled,
        },
        "abort_controls": {
            "sigint": True,
            "abort_file": str(abort_file) if abort_file else None,
        },
        "attestations": attestations,
        "bindings": {
            "safety_profile": str(safety_profile) if safety_profile else None,
            "safety_profile_sha256": _sha256_file(safety_profile),
            "readiness_report": str(readiness_report) if readiness_report else None,
            "readiness_report_sha256": _sha256_file(readiness_report),
            "approval": str(approval) if approval else None,
            "approval_sha256": _sha256_file(approval),
        },
    }


def build_arming_markdown(manifest: dict[str, Any]) -> str:
    limits = manifest.get("limits", {})
    b = manifest.get("bindings", {})
    ac = manifest.get("abort_controls", {})
    return (
        "# Real Session Arming Manifest\n\n"
        f"- Session: {manifest.get('session_id')}\n"
        f"- Armed at: {manifest.get('armed_at')}\n"
        f"- Operator: {manifest.get('operator')}\n"
        f"- Policy: {manifest.get('policy_path')}\n"
        f"- Approval id: {manifest.get('approval_id')}\n"
        f"- Limits: max_steps={limits.get('max_steps')} "
        f"duration_seconds={limits.get('duration_seconds')} fps={limits.get('fps')} "
        f"deadman={limits.get('deadman_timeout_s')}s "
        f"(enabled={limits.get('deadman_enabled')})\n"
        f"- Abort controls: SIGINT={ac.get('sigint')} abort_file={ac.get('abort_file')}\n"
        f"- readiness_report_sha256: {b.get('readiness_report_sha256')}\n"
        f"- approval_sha256: {b.get('approval_sha256')}\n\n"
        "This manifest records the armed envelope only. It does not prove "
        "physical safety or real-world success.\n"
    )
