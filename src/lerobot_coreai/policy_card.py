# policy_card.py — generate honest policy cards from verified evidence (v1.2.3).
#
# A policy card is generated deterministically from already-verified artifacts,
# never written by hand. Sources are verified before use (benchmark checksums,
# indexed-artifact integrity) and scanned for overclaims; either failure aborts
# card generation. The card always carries the mandatory non-claims. Proves the
# card was generated from verified evidence — never physical safety, task
# success, training support, native registry, or actuation authorization.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from . import policy_card_templates as T
from .release_governance import _find_true_claims

POLICY_CARD_REPORT_SCHEMA_VERSION = "lerobot-coreai.policy_card_report.v0"

# Canonical report filenames inside a bridge benchmark bundle.
_BUNDLE_REPORTS = {
    "compat": "reports/lerobot_compatibility_report.json",
    "bridge": "reports/lerobot_bridge_report.json",
    "registry": "reports/lerobot_registry_report.json",
    "eval_v2": "reports/eval_v2_report.json",
    "obs_bridge": "reports/obs_bridge_report.json",
}


class PolicyCardError(Exception):
    """Raised when a card cannot be generated (fail-closed)."""


@dataclass
class PolicyCardInputs:
    benchmark_bundle: Path | None = None
    compat_report: Path | None = None
    bridge_report: Path | None = None
    registry_report: Path | None = None
    eval_v2_report: Path | None = None
    obs_bridge_report: Path | None = None
    provenance: Path | None = None
    signature: Path | None = None
    release_check: Path | None = None
    release_channel: str | None = None
    # Index mode.
    artifact_index: Path | None = None
    artifact_id: str | None = None


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _resolve_from_index(inputs: PolicyCardInputs) -> tuple[Path, str | None, str | None]:
    """Resolve the benchmark bundle dir + release channel from an index entry."""
    from .artifact_index import list_entries, verify_index
    entries = list_entries(inputs.artifact_index)
    match = next((e for e in entries if e.get("artifact_id") == inputs.artifact_id), None)
    if match is None:
        raise PolicyCardError(f"artifact_id {inputs.artifact_id!r} not found in index.")
    # The indexed artifact must still verify.
    vres = verify_index(inputs.artifact_index)
    if not vres.ok:
        raise PolicyCardError("indexed artifact failed verification; refusing to card it.")
    return Path(match["artifact_dir"]), match.get("release_channel"), \
        match.get("artifact_id")


def generate_policy_card(inputs: PolicyCardInputs) -> tuple[str, dict[str, Any]]:
    """Return (markdown_card, report). Fail-closed on tamper/overclaim."""
    source_mode = "artifact_index" if inputs.artifact_index else "direct"
    artifact_id = inputs.artifact_id
    release_channel = inputs.release_channel

    bundle_dir = inputs.benchmark_bundle
    if inputs.artifact_index:
        if not inputs.artifact_id:
            raise PolicyCardError("--artifact-index requires --artifact-id.")
        bundle_dir, release_channel, artifact_id = _resolve_from_index(inputs)

    # Verify the benchmark bundle before using any of its reports.
    reports: dict[str, dict[str, Any] | None] = {}
    manifest = None
    if bundle_dir:
        from .benchmark_pack import verify_bridge_benchmark
        vres = verify_bridge_benchmark(bundle_dir)
        if not vres.ok:
            raise PolicyCardError(
                "benchmark bundle failed verification; refusing to generate a card.")
        bundle_dir = Path(bundle_dir)
        manifest = _load(bundle_dir / "benchmark_manifest.json")
        for slot, rel in _BUNDLE_REPORTS.items():
            reports[slot] = _load(bundle_dir / rel)
    else:
        reports = {
            "compat": _load(inputs.compat_report),
            "bridge": _load(inputs.bridge_report),
            "registry": _load(inputs.registry_report),
            "eval_v2": _load(inputs.eval_v2_report),
            "obs_bridge": _load(inputs.obs_bridge_report),
        }
        if not any(reports.values()):
            raise PolicyCardError(
                "no source reports provided; pass a benchmark bundle, an "
                "artifact-index entry, or direct report paths.")

    provenance = _load(inputs.provenance)
    signature = _load(inputs.signature)
    release_check = _load(inputs.release_check)

    # Fail-closed on any overclaim across all consulted sources.
    overclaims: list[str] = []
    for data in list(reports.values()) + [manifest, provenance, release_check]:
        if data:
            overclaims += _find_true_claims(data)
    if overclaims:
        raise PolicyCardError(
            f"refusing to card overclaiming evidence: {sorted(set(overclaims))}")

    # Identity hints.
    policy_path = (manifest or {}).get("policy_path") if manifest else None
    dataset_repo_id = (manifest or {}).get("dataset_repo_id") if manifest else None
    bridge = reports.get("bridge") or {}
    policy_path = policy_path or bridge.get("policy_path") \
        or (reports.get("compat") or {}).get("policy_path")
    robot_type = bridge.get("robot_type")
    ev2 = reports.get("eval_v2") or {}
    dataset_repo_id = dataset_repo_id or ev2.get("dataset_repo_id")

    sections: list[tuple[str, str]] = [
        ("what_this_policy_is", T.section_what(policy_path, robot_type)),
        ("runtime", T.section_runtime()),
        ("how_to_run", T.section_how_to_run()),
        ("compatibility_evidence", T.section_compat(reports.get("compat"))),
        ("bridge_evidence", T.section_bridge(reports.get("bridge"))),
        ("registry_evidence", T.section_registry(reports.get("registry"))),
        ("feature_mapping_summary", T.section_feature_mapping(reports.get("eval_v2"))),
        ("eval_v2_summary", T.section_eval_v2(reports.get("eval_v2"))),
        ("observation_pipeline_summary", T.section_obs_bridge(reports.get("obs_bridge"))),
        ("benchmark_pack_summary", T.section_benchmark(manifest)),
        ("trust_status", T.section_trust(
            provenance=provenance, signature=signature, release_check=release_check,
            release_channel=release_channel)),
        ("non_claims", T.section_limitations_and_non_claims()),
    ]
    written = [name for name, body in sections if body]
    title = f"# {policy_path or 'CoreAI policy'} — Policy Card\n\n"
    card = title + "\n".join(body for _n, body in sections if body)

    report = {
        "schema_version": POLICY_CARD_REPORT_SCHEMA_VERSION,
        "ok": True,
        "artifact_id": artifact_id,
        "source_verified": True,
        "source_mode": source_mode,
        "policy_path": policy_path,
        "dataset_repo_id": dataset_repo_id,
        "lerobot_coreai_version": __version__,
        "sections_written": written,
        "input_reports": {k: _BUNDLE_REPORTS.get(k) for k, v in reports.items() if v},
        "claims": {
            "proves_policy_card_generated_from_verified_evidence": True,
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
            "supports_training": False,
            "native_upstream_registry": False,
        },
    }
    return card, report
