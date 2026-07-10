# provenance.py — provenance manifests for publishable artifacts (v1.2.0).
#
# Checksums say "these bytes weren't altered"; provenance says "this is what the
# artifact is, where it came from, and what evidence backs it". Provenance is the
# payload a signature commits to. It proves origin/integrity metadata only — never
# task success, physical safety, or actuation authorization.

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__

PROVENANCE_SCHEMA_VERSION = "lerobot-coreai.provenance.v0"

# Files whose hashes anchor the provenance (when present in the artifact dir).
_ANCHOR_FILES = ("benchmark_manifest.json", "checksums.json")


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_provenance(artifact_dir: Path, artifact_type: str, *,
                     created_at: str | None = None) -> dict[str, Any]:
    """Build a provenance manifest for an artifact directory."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.is_dir():
        from .errors import CoreAIPolicyError
        raise CoreAIPolicyError(f"artifact dir not found: {artifact_dir}")

    artifact_hashes: dict[str, str] = {}
    for name in _ANCHOR_FILES:
        p = artifact_dir / name
        if p.is_file():
            artifact_hashes[name] = sha256_file(p)

    source_reports: dict[str, str] = {}
    manifest_path = artifact_dir / "benchmark_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            source_reports = dict(manifest.get("reports", {}))
        except Exception:
            source_reports = {}

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "artifact_dir": str(artifact_dir),
        "created_at": created_at or _now_iso(),
        "lerobot_coreai_version": __version__,
        "source_reports": source_reports,
        "artifact_hashes": artifact_hashes,
        "claims": {
            "proves_artifact_integrity": True,
            "proves_artifact_origin_metadata": True,
            "proves_task_success": False,
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
        },
    }


def build_provenance_markdown(provenance: dict[str, Any]) -> str:
    lines = [
        "# Artifact Provenance",
        "",
        f"- Artifact type: {provenance.get('artifact_type')}",
        f"- Artifact dir: {provenance.get('artifact_dir')}",
        f"- Created at: {provenance.get('created_at')}",
        f"- lerobot-coreai: {provenance.get('lerobot_coreai_version')}",
        "",
        "## Anchored hashes",
    ]
    for name, h in provenance.get("artifact_hashes", {}).items():
        lines.append(f"- `{name}`: {h}")
    if provenance.get("source_reports"):
        lines += ["", "## Source reports"]
        for slot, rel in provenance["source_reports"].items():
            lines.append(f"- `{slot}` → `{rel}`")
    lines += [
        "",
        "Proves artifact integrity + origin metadata only — not task success, "
        "not physical safety, and authorizes no robot actuation.",
        "",
    ]
    return "\n".join(lines)
