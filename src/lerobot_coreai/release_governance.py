# release_governance.py — release channel policy + release-check (v1.2.1).
#
# Not every verifiable artifact should be publishable on every channel. A release
# policy (per channel) decides: which reports are required, whether a valid
# signature is required, whether overclaims / raw secrets / real-session /
# external-http artifacts are allowed. Fail-closed: anything not explicitly
# permitted blocks the release. Proves publishability-under-policy only — never
# physical safety or actuation authorization.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__

RELEASE_POLICY_SCHEMA_VERSION = "lerobot-coreai.release_policy.v0"
RELEASE_REPORT_SCHEMA_VERSION = "lerobot-coreai.release_check_report.v0"

# Claim keys that must never be true in a released artifact.
_FORBIDDEN_CLAIMS = {
    "proves_physical_safety", "proves_real_world_safety", "physical_safety_proof",
    "supports_physical_safety", "authorizes_robot_actuation",
    "authorizes_unrestricted_real_world_actuation", "unrestricted_actuation",
    "native_upstream_registry", "supports_training",
}

# Files that mark an artifact as carrying real-session / external-http evidence.
_REAL_SESSION_FILES = ("real_report.json", "real_session.json", "real_trace.jsonl")
_GUARDED_REAL_REQUIRED = ("approval", "readiness", "verify_real_session")

CHANNELS = ("dev", "internal", "public-demo", "research", "guarded-real-evidence")


@dataclass
class ReleasePolicy:
    channel: str
    required_reports: list[str] = field(default_factory=list)
    require_signature: bool = False
    require_no_overclaims: bool = True
    require_no_raw_secrets: bool = True
    allow_real_session_artifacts: bool = True
    allow_external_http_artifacts: bool = True
    require_guarded_real_evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RELEASE_POLICY_SCHEMA_VERSION,
            "channel": self.channel,
            "required_reports": self.required_reports,
            "require_signature": self.require_signature,
            "require_no_overclaims": self.require_no_overclaims,
            "require_no_raw_secrets": self.require_no_raw_secrets,
            "allow_real_session_artifacts": self.allow_real_session_artifacts,
            "allow_external_http_artifacts": self.allow_external_http_artifacts,
            "require_guarded_real_evidence": self.require_guarded_real_evidence,
        }


# Canonical filenames as written into a bridge benchmark pack (see benchmark_pack).
_BRIDGE_REPORTS = [
    "lerobot_compatibility_report.json", "lerobot_bridge_report.json",
    "feature_mapping.json", "eval_v2_report.json", "obs_bridge_report.json",
]


def default_policy(channel: str) -> ReleasePolicy:
    """Built-in policy per channel. Public channels are the strictest."""
    if channel == "dev":
        return ReleasePolicy(channel, require_no_overclaims=False,
                             require_no_raw_secrets=False)
    if channel == "internal":
        return ReleasePolicy(channel)
    if channel == "public-demo":
        return ReleasePolicy(
            channel, required_reports=_BRIDGE_REPORTS, require_signature=True,
            require_no_overclaims=True, require_no_raw_secrets=True,
            allow_real_session_artifacts=False, allow_external_http_artifacts=False)
    if channel == "research":
        return ReleasePolicy(channel, require_signature=True,
                             require_no_overclaims=True)
    if channel == "guarded-real-evidence":
        return ReleasePolicy(
            channel, require_signature=True, require_no_overclaims=True,
            require_no_raw_secrets=True, require_guarded_real_evidence=True)
    raise ValueError(f"unknown channel {channel!r}; choose from {CHANNELS}")


def load_release_policy(path: Path) -> ReleasePolicy:
    data = json.loads(Path(path).read_text())
    return ReleasePolicy(
        channel=data.get("channel", "custom"),
        required_reports=data.get("required_reports", []),
        require_signature=data.get("require_signature", False),
        require_no_overclaims=data.get("require_no_overclaims", True),
        require_no_raw_secrets=data.get("require_no_raw_secrets", True),
        allow_real_session_artifacts=data.get("allow_real_session_artifacts", True),
        allow_external_http_artifacts=data.get("allow_external_http_artifacts", True),
        require_guarded_real_evidence=data.get("require_guarded_real_evidence", False))


def _find_true_claims(obj: Any) -> list[str]:
    found: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _FORBIDDEN_CLAIMS and v is True:
                    found.append(k)
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)

    _walk(obj)
    return found


# Heuristics for a leaked secret: a token/secret/authorization value that is not
# redacted and not a sha256 fingerprint/hash.
_SECRET_KEYS = re.compile(r"(token|secret|password|api[_-]?key|private[_-]?key)", re.I)


def _find_raw_secrets(obj: Any) -> list[str]:
    found: list[str] = []

    def _ok(v: str) -> bool:
        return v in ("<redacted>", "") or v.startswith("sha256:") or v.endswith("_env") \
            or v.isupper()  # env var *names* are allowed, values are not

    def _walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                kp = f"{path}.{k}" if path else k
                if isinstance(v, str) and _SECRET_KEYS.search(k) and not _ok(v):
                    # allow explicit env-name / fingerprint fields
                    if not (k.endswith("_env") or k.endswith("fingerprint")
                            or k.endswith("sha256_prefix") or k.endswith("token_source")):
                        found.append(kp)
                _walk(v, kp)
        elif isinstance(node, list):
            for i, x in enumerate(node):
                _walk(x, f"{path}[{i}]")
        elif isinstance(node, str):
            if "Bearer " in node or "-----BEGIN" in node:
                found.append(path or "<value>")

    _walk(obj)
    return found


def _iter_report_files(artifact_dir: Path):
    for p in sorted(artifact_dir.rglob("*.json")):
        try:
            yield p, json.loads(p.read_text())
        except Exception:
            continue


def evaluate_release(
    artifact_dir: Path, *, artifact_type: str, policy: ReleasePolicy,
    signature: Path | None = None, provenance: Path | None = None,
    trust_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an artifact dir against a release policy. Fail-closed."""
    artifact_dir = Path(artifact_dir)
    checks: list[dict[str, Any]] = []

    def _c(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    # Required reports present (searched recursively).
    present_names = {p.name for p in artifact_dir.rglob("*.json")}
    missing = [r for r in policy.required_reports if r not in present_names]
    _c("required_reports_present", not missing,
       "" if not missing else f"missing: {missing}")

    # Signature.
    if policy.require_signature:
        if signature and provenance and signature.is_file() and provenance.is_file():
            from .trust_policy import verify_signed_artifact
            res = verify_signed_artifact(artifact_dir, provenance, signature,
                                         trust_policy=trust_policy)
            _c("signature_valid", res.ok,
               "" if res.ok else "signature verification failed")
        else:
            _c("signature_valid", False, "channel requires a signature; none provided")

    # Overclaims.
    if policy.require_no_overclaims:
        overclaims = []
        for _p, data in _iter_report_files(artifact_dir):
            overclaims += _find_true_claims(data)
        _c("no_overclaims", not overclaims,
           "" if not overclaims else f"overclaims: {sorted(set(overclaims))}")

    # Raw secrets.
    if policy.require_no_raw_secrets:
        leaks = []
        for p, data in _iter_report_files(artifact_dir):
            hits = _find_raw_secrets(data)
            leaks += [f"{p.name}:{h}" for h in hits]
        _c("no_raw_secrets", not leaks,
           "" if not leaks else f"possible secrets: {leaks}")

    # Real-session artifacts.
    if not policy.allow_real_session_artifacts:
        real = [f for f in _REAL_SESSION_FILES
                if any(p.name == f for p in artifact_dir.rglob(f))]
        _c("no_real_session_artifacts", not real,
           "" if not real else f"real-session files present: {real}")

    # External-http artifacts.
    if not policy.allow_external_http_artifacts:
        ext = []
        for p, data in _iter_report_files(artifact_dir):
            if "external-http" in json.dumps(data) or "external_http" in json.dumps(data):
                ext.append(p.name)
        _c("no_external_http_artifacts", not ext,
           "" if not ext else f"external-http referenced in: {sorted(set(ext))}")

    # Guarded-real evidence must be complete.
    if policy.require_guarded_real_evidence:
        names_blob = " ".join(present_names)
        has_approval = "approval" in names_blob
        has_readiness = "readiness" in names_blob
        has_verify = any("verification" in n or "verify_real" in n for n in present_names)
        ok = has_approval and has_readiness and has_verify
        _c("guarded_real_evidence_complete", ok,
           "" if ok else "need approval + readiness + verify-real-session artifacts")

    ok = all(c["passed"] for c in checks)
    return {
        "schema_version": RELEASE_REPORT_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "channel": policy.channel,
        "artifact_type": artifact_type,
        "artifact_dir": str(artifact_dir),
        "policy": policy.to_dict(),
        "checks": checks,
        "claims": {
            "proves_release_policy_satisfied": ok,
            "proves_task_success": False,
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
        },
    }


def build_release_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Release Check",
        "",
        f"- Channel: {report.get('channel')}",
        f"- Artifact type: {report.get('artifact_type')}",
        f"- OK: {report.get('ok')}",
        "",
        "## Checks",
    ]
    for c in report.get("checks", []):
        mark = "✅" if c["passed"] else "❌"
        detail = f" — {c['detail']}" if c.get("detail") else ""
        lines.append(f"- {mark} `{c['name']}`{detail}")
    lines += [
        "",
        "Proves the artifact satisfies the channel release policy — not task "
        "success or physical safety, and authorizes no actuation.",
        "",
    ]
    return "\n".join(lines)
