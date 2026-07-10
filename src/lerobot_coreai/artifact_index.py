# artifact_index.py — local registry for signed/verified artifacts (v1.2.2).
#
# Once artifacts are verifiable and signable, they need to be findable. This is a
# local index: add a verified artifact (checksums + optional signature +
# release-check status), then list/find/verify. Fail-closed: an add is refused if
# the artifact is tampered, overclaims, leaks a secret, or (when a trust policy is
# given) is signed by an untrusted key. Proves discovery/integrity only — never
# physical safety or actuation authorization.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .release_governance import _find_raw_secrets, _find_true_claims

ARTIFACT_INDEX_SCHEMA_VERSION = "lerobot-coreai.artifact_index.v0"
ARTIFACT_INDEX_ENTRY_SCHEMA_VERSION = "lerobot-coreai.artifact_index_entry.v0"


class ArtifactIndexError(Exception):
    """Raised when an index operation must fail closed."""


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _index_root(index_dir: Path) -> Path:
    return Path(index_dir) / "index.json"


def _entries_dir(index_dir: Path) -> Path:
    return Path(index_dir) / "entries"


def init_index(index_dir: Path) -> dict[str, Any]:
    """Initialize an empty index directory (idempotent)."""
    index_dir = Path(index_dir)
    _entries_dir(index_dir).mkdir(parents=True, exist_ok=True)
    root_path = _index_root(index_dir)
    if root_path.is_file():
        return json.loads(root_path.read_text())
    root = {
        "schema_version": ARTIFACT_INDEX_SCHEMA_VERSION,
        "created_at": _now_iso(),
        "entries": [],
        "claims": {
            "proves_artifact_discovery": True,
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
        },
    }
    root_path.write_text(json.dumps(root, indent=2))
    return root


def _load_root(index_dir: Path) -> dict[str, Any]:
    root_path = _index_root(index_dir)
    if not root_path.is_file():
        raise ArtifactIndexError(
            f"no index at {index_dir}; run 'artifact-index init' first.")
    return json.loads(root_path.read_text())


def _verify_checksums(artifact_dir: Path) -> list[str]:
    """Return a list of tampered/missing files per checksums.json (empty = clean)."""
    checks_path = artifact_dir / "checksums.json"
    if not checks_path.is_file():
        return []  # nothing to verify against
    tampered: list[str] = []
    checksums = json.loads(checks_path.read_text())
    for rel, expected in checksums.items():
        fp = artifact_dir / rel
        actual = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else None
        if actual != expected:
            tampered.append(rel)
    return tampered


def _scan_reports(artifact_dir: Path):
    for p in sorted(artifact_dir.rglob("*.json")):
        try:
            yield p, json.loads(p.read_text())
        except Exception:
            continue


@dataclass
class AddResult:
    entry: dict[str, Any]
    entry_path: Path


def add_artifact(
    index_dir: Path, artifact_dir: Path, artifact_type: str, *,
    release_channel: str | None = None, policy_path: str | None = None,
    dataset_repo_id: str | None = None, signature: Path | None = None,
    provenance: Path | None = None, trust_policy: dict[str, Any] | None = None,
    release_check_report: Path | None = None, created_at: str | None = None,
    force: bool = False,
) -> AddResult:
    """Verify and add an artifact to the index. Fail-closed."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        raise ArtifactIndexError(f"artifact dir not found: {artifact_dir}")
    _load_root(index_dir)  # ensure initialized

    manifest_path = artifact_dir / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    policy_path = policy_path or manifest.get("policy_path")
    dataset_repo_id = dataset_repo_id or manifest.get("dataset_repo_id")

    # Fail-closed checks before indexing.
    tampered = _verify_checksums(artifact_dir)
    if tampered:
        raise ArtifactIndexError(f"artifact is tampered (checksum mismatch): {tampered}")

    overclaims, secrets = [], []
    for p, data in _scan_reports(artifact_dir):
        overclaims += _find_true_claims(data)
        secrets += [f"{p.name}:{h}" for h in _find_raw_secrets(data)]
    if overclaims:
        raise ArtifactIndexError(
            f"refusing to index an overclaiming artifact: {sorted(set(overclaims))}")
    if secrets:
        raise ArtifactIndexError(
            f"refusing to index an artifact with raw secrets: {secrets}")

    # Signature (optional). signature_verified is only ever true after a real check.
    signature_verified = False
    signature_fingerprint = None
    if signature and provenance and Path(signature).is_file() and Path(provenance).is_file():
        from .trust_policy import verify_signed_artifact
        res = verify_signed_artifact(artifact_dir, Path(provenance), Path(signature),
                                     trust_policy=trust_policy)
        signature_verified = res.ok
        try:
            signature_fingerprint = json.loads(Path(signature).read_text())\
                .get("signer", {}).get("key_fingerprint")
        except Exception:
            signature_fingerprint = None

    release_check_passed = None
    if release_check_report and Path(release_check_report).is_file():
        try:
            release_check_passed = bool(
                json.loads(Path(release_check_report).read_text()).get("ok"))
        except Exception:
            release_check_passed = None

    created_at = created_at or _now_iso()
    base_id = artifact_dir.name
    artifact_id = f"{base_id}@{created_at.replace(':', '-')}"

    entry = {
        "schema_version": ARTIFACT_INDEX_ENTRY_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "artifact_dir": str(artifact_dir),
        "policy_path": policy_path,
        "dataset_repo_id": dataset_repo_id,
        "lerobot_coreai_version": __version__,
        "release_channel": release_channel,
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else None,
        "provenance_sha256": _sha256_file(Path(provenance)) if provenance and Path(provenance).is_file() else None,
        "signature_fingerprint": signature_fingerprint,
        "signature_verified": signature_verified,
        "release_check_passed": release_check_passed,
        "created_at": created_at,
        "claims": {
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
            "native_upstream_registry": False,
        },
    }

    entry_path = _entries_dir(index_dir) / f"{artifact_id}.json"
    if entry_path.exists() and not force:
        raise ArtifactIndexError(
            f"artifact_id {artifact_id!r} already indexed; refusing to overwrite "
            "(pass force=True to replace).")
    entry_path.write_text(json.dumps(entry, indent=2))

    # Update the root's entry list.
    root = _load_root(index_dir)
    rel = f"entries/{artifact_id}.json"
    if rel not in root["entries"]:
        root["entries"].append(rel)
        root["entries"] = sorted(set(root["entries"]))
        _index_root(index_dir).write_text(json.dumps(root, indent=2))
    return AddResult(entry=entry, entry_path=entry_path)


def list_entries(index_dir: Path) -> list[dict[str, Any]]:
    root = _load_root(index_dir)
    out = []
    for rel in root.get("entries", []):
        p = Path(index_dir) / rel
        if p.is_file():
            out.append(json.loads(p.read_text()))
    return out


def find_entries(index_dir: Path, *, policy_path: str | None = None,
                 dataset_repo_id: str | None = None, artifact_type: str | None = None,
                 release_channel: str | None = None) -> list[dict[str, Any]]:
    def _match(e):
        return ((policy_path is None or e.get("policy_path") == policy_path)
                and (dataset_repo_id is None or e.get("dataset_repo_id") == dataset_repo_id)
                and (artifact_type is None or e.get("artifact_type") == artifact_type)
                and (release_channel is None or e.get("release_channel") == release_channel))
    return [e for e in list_entries(index_dir) if _match(e)]


@dataclass
class IndexVerifyResult:
    ok: bool
    checks: list[dict[str, Any]]


def verify_index(index_dir: Path) -> IndexVerifyResult:
    """Verify every indexed artifact still exists, matches its manifest hash, and
    that no indexed artifact has become tampered or started overclaiming."""
    checks: list[dict[str, Any]] = []

    def _c(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        entries = list_entries(index_dir)
    except ArtifactIndexError as e:
        _c("index_readable", False, str(e))
        return IndexVerifyResult(ok=False, checks=checks)

    for e in entries:
        aid = e["artifact_id"]
        adir = Path(e["artifact_dir"])
        if not adir.is_dir():
            _c(f"{aid}:artifact_dir_exists", False, str(adir))
            continue
        _c(f"{aid}:artifact_dir_exists", True)
        # Manifest hash still matches.
        mpath = adir / "benchmark_manifest.json"
        if e.get("manifest_sha256"):
            match = mpath.is_file() and _sha256_file(mpath) == e["manifest_sha256"]
            _c(f"{aid}:manifest_hash_matches", match)
        # No new tamper / overclaim.
        tampered = _verify_checksums(adir)
        _c(f"{aid}:checksums_valid", not tampered,
           "" if not tampered else f"tampered: {tampered}")
        overclaims = []
        for _p, data in _scan_reports(adir):
            overclaims += _find_true_claims(data)
        _c(f"{aid}:no_overclaims", not overclaims,
           "" if not overclaims else f"{sorted(set(overclaims))}")

    # Empty index (no checks) verifies vacuously.
    ok = all(c["passed"] for c in checks)
    return IndexVerifyResult(ok=ok, checks=checks)
