# benchmark_pack.py — reproducible bridge benchmark packs (v1.1.7).
#
# Bundles the bridge/compat/registry/eval-v2/obs-bridge reports into a single
# reproducibility pack with per-file SHA256 checksums, an auto-generated README,
# and a verifier that detects tampering. Fail-closed on overclaim: a report that
# claims physical safety, task success, or actuation authorization is refused at
# packaging time and flagged at verify time. Software artifacts only.

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__

BRIDGE_BENCHMARK_SCHEMA_VERSION = "lerobot-coreai.bridge_benchmark_pack.v0"

# Boolean claim keys that must never be True in an included report.
_BANNED_TRUE_CLAIMS = {
    "proves_physical_safety",
    "proves_real_world_safety",
    "physical_safety_proof",
    "supports_physical_safety",
    "authorizes_robot_actuation",
    "authorizes_unrestricted_real_world_actuation",
    "unrestricted_actuation",
    "proves_task_success",
    "supports_training",
    "native_upstream_registry",
    "native_registry",
    "upstream_native",
}

# The report slots a pack may carry. Value = canonical filename in the bundle.
_REPORT_SLOTS = {
    "compat": "lerobot_compatibility_report.json",
    "bridge": "lerobot_bridge_report.json",
    "registry": "lerobot_registry_report.json",
    "feature_mapping": "feature_mapping.json",
    "eval_v2": "eval_v2_report.json",
    "obs_bridge": "obs_bridge_report.json",
}


@dataclass
class BenchmarkInputs:
    compat: Path | None = None
    bridge: Path | None = None
    registry: Path | None = None
    feature_mapping: Path | None = None
    eval_v2: Path | None = None
    obs_bridge: Path | None = None
    policy_path: str | None = None
    dataset_repo_id: str | None = None

    def slots(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for slot in _REPORT_SLOTS:
            p = getattr(self, slot)
            if p is not None:
                out[slot] = Path(p)
        return out


@dataclass
class BenchmarkResult:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    manifest: dict[str, Any] | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _find_overclaims(obj: Any) -> list[str]:
    """Recursively find banned claim keys set to True."""
    found: list[str] = []

    def _walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in _BANNED_TRUE_CLAIMS and v is True:
                    found.append(k)
                _walk(v)
        elif isinstance(node, list):
            for x in node:
                _walk(x)

    _walk(obj)
    return found


class BenchmarkError(Exception):
    """Raised when a benchmark pack cannot be built (fail-closed)."""


def package_bridge_benchmark(inputs: BenchmarkInputs, output_dir: Path) -> dict[str, Any]:
    """Assemble a bridge benchmark pack. Fail-closed on missing/overclaiming reports."""
    slots = inputs.slots()
    if not slots:
        raise BenchmarkError("no reports provided; a benchmark pack needs at least one.")

    output_dir = Path(output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    checksums: dict[str, str] = {}
    reports_map: dict[str, str] = {}
    policy_path = inputs.policy_path
    dataset_repo_id = inputs.dataset_repo_id

    for slot, src in slots.items():
        if not src.is_file():
            raise BenchmarkError(f"report for {slot!r} not found: {src}")
        data = json.loads(src.read_text())
        overclaims = _find_overclaims(data)
        if overclaims:
            raise BenchmarkError(
                f"refusing to package {slot!r}: overclaiming keys set true "
                f"{sorted(set(overclaims))}.")
        # Pull identity hints if not supplied explicitly.
        policy_path = policy_path or data.get("policy_path")
        dataset_repo_id = dataset_repo_id or data.get("dataset_repo_id")
        rel = f"reports/{_REPORT_SLOTS[slot]}"
        shutil.copyfile(src, output_dir / rel)
        checksums[rel] = _sha256(output_dir / rel)
        reports_map[slot] = rel

    manifest = {
        "schema_version": BRIDGE_BENCHMARK_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "bundle_type": "bridge_benchmark",
        "policy_path": policy_path,
        "dataset_repo_id": dataset_repo_id,
        "reports": reports_map,
        "claims": {
            "proves_bridge_benchmark_reproducibility": True,
            "proves_task_success": False,
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
        },
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode()
    (output_dir / "benchmark_manifest.json").write_bytes(manifest_bytes)
    checksums["benchmark_manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2))
    (output_dir / "README.md").write_text(build_benchmark_readme(manifest, checksums))
    return manifest


def build_benchmark_readme(manifest: dict[str, Any], checksums: dict[str, str]) -> str:
    lines = [
        "# Bridge Benchmark Pack",
        "",
        f"- lerobot-coreai: {manifest.get('lerobot_coreai_version')}",
        f"- policy: {manifest.get('policy_path')}",
        f"- dataset: {manifest.get('dataset_repo_id')}",
        "",
        "## Included reports",
    ]
    for slot, rel in manifest.get("reports", {}).items():
        lines.append(f"- `{slot}` → `{rel}`")
    lines += [
        "",
        "## Verify",
        "",
        "```bash",
        "lerobot-coreai verify-bridge-benchmark --bundle-dir .",
        "```",
        "",
        "Checksums in `checksums.json` bind every file; `verify` recomputes them "
        "and fails on any mismatch (tamper detection).",
        "",
        "## Scope",
        "",
        "This pack bundles **software** compatibility/bridge/eval reports for "
        "reproducibility. It does **not** prove task success or physical safety, "
        "and authorizes no robot actuation.",
        "",
    ]
    return "\n".join(lines)


def verify_bridge_benchmark(bundle_dir: Path) -> BenchmarkResult:
    """Verify a benchmark pack: presence, checksums (tamper), and no overclaim."""
    bundle_dir = Path(bundle_dir)
    checks: list[dict[str, Any]] = []

    def _c(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    mpath = bundle_dir / "benchmark_manifest.json"
    cpath = bundle_dir / "checksums.json"
    if not mpath.is_file() or not cpath.is_file():
        _c("manifest_and_checksums_present", False, "manifest or checksums missing")
        return BenchmarkResult(ok=False, checks=checks)
    _c("manifest_and_checksums_present", True)

    manifest = json.loads(mpath.read_text())
    checksums = json.loads(cpath.read_text())

    # Schema.
    try:
        import jsonschema
        from importlib.resources import files
        schema = json.loads(files("lerobot_coreai.schemas").joinpath(
            "bridge-benchmark-pack.schema.json").read_text())
        jsonschema.validate(manifest, schema)
        _c("manifest_schema_valid", True)
    except Exception as e:
        _c("manifest_schema_valid", False, getattr(e, "message", str(e)))

    # Every listed report exists.
    reports = manifest.get("reports", {})
    missing = [rel for rel in reports.values() if not (bundle_dir / rel).is_file()]
    _c("all_reports_present", not missing,
       "" if not missing else f"missing: {missing}")

    # Checksums match (tamper detection). Every recorded file, and the manifest.
    tampered = []
    for rel, expected in checksums.items():
        fp = bundle_dir / rel
        if not fp.is_file() or _sha256(fp) != expected:
            tampered.append(rel)
    _c("checksums_match", not tampered,
       "" if not tampered else f"tampered/missing: {tampered}")

    # No overclaim in any bundled report.
    overclaims: list[str] = []
    for rel in reports.values():
        fp = bundle_dir / rel
        if fp.is_file():
            try:
                overclaims += _find_overclaims(json.loads(fp.read_text()))
            except Exception:
                pass
    _c("no_report_overclaim", not overclaims,
       "" if not overclaims else f"overclaims: {sorted(set(overclaims))}")

    # Manifest itself must not claim safety/task.
    mc = manifest.get("claims", {})
    honest = (mc.get("proves_physical_safety") is False
              and mc.get("proves_task_success") is False
              and mc.get("authorizes_robot_actuation") is False)
    _c("manifest_claims_honest", honest)

    ok = all(c["passed"] for c in checks)
    return BenchmarkResult(ok=ok, checks=checks, manifest=manifest)
