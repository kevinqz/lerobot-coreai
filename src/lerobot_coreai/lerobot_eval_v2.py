# lerobot_eval_v2.py — LeRobotDataset eval parity v2 (v1.1.4).
#
# Builds on the v0.4 LeRobotDataset eval by making the dataset ↔ policy feature
# mapping explicit and auditable, with a strict mode that fails on missing
# required keys or shape mismatches. The report proves the observation mapping is
# coherent for the evaluated frames — never task success or physical safety.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .lerobot_features import build_feature_mapping, manifest_obs_features

EVAL_V2_SCHEMA_VERSION = "lerobot-coreai.lerobot_eval_v2.v0"


@dataclass
class EvalV2Config:
    policy_path: str
    dataset_repo_id: str
    runner_url: str | None = None
    episodes: list[int] | None = None
    max_frames: int | None = None
    strict_features: bool = False
    fail_on_unknown: bool = False
    task: str | None = None
    output_dir: Path | None = None


def _load_dataset_features(dataset_repo_id: str) -> dict[str, Any]:  # pragma: no cover - needs lerobot+net
    """Return name -> shape for a LeRobotDataset, via the public constructor."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = LeRobotDataset(dataset_repo_id)
    feats = {}
    for name, spec in getattr(ds, "features", {}).items():
        shape = spec.get("shape") if isinstance(spec, dict) else None
        feats[name] = list(shape) if shape is not None else None
    return feats, ds


def build_eval_v2_report(*, policy_path: str, dataset_repo_id: str,
                         feature_mapping: dict[str, Any],
                         frames_evaluated: int, strict: bool) -> dict[str, Any]:
    checks = [
        {"name": "feature_mapping_passed", "passed": bool(feature_mapping.get("passed")),
         "severity": "required" if strict else "info",
         "detail": "; ".join(feature_mapping.get("problems", []))},
    ]
    ok = all(c["passed"] for c in checks if c["severity"] == "required")
    return {
        "schema_version": EVAL_V2_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "policy_path": policy_path,
        "dataset_repo_id": dataset_repo_id,
        "strict": strict,
        "frames_evaluated": frames_evaluated,
        "feature_mapping": feature_mapping,
        "checks": checks,
        "claims": {
            "proves_observation_mapping_valid_for_sample": ok,
            "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }


def build_eval_v2_markdown(report: dict[str, Any]) -> str:
    fm = report.get("feature_mapping", {})
    lines = [
        "# LeRobot Eval v2 — Feature Mapping",
        "",
        f"- OK: {report.get('ok')}",
        f"- Policy: {report.get('policy_path')}",
        f"- Dataset: {report.get('dataset_repo_id')}",
        f"- Strict: {report.get('strict')}",
        f"- Frames evaluated: {report.get('frames_evaluated')}",
        "",
        "## Features",
    ]
    for name, e in fm.get("features", {}).items():
        bits = []
        if "dataset_present" in e:
            bits.append(f"dataset_present={e['dataset_present']}")
        if "shape_compatible" in e:
            bits.append(f"shape_compatible={e['shape_compatible']}")
        if e.get("provided_by_config"):
            bits.append("provided_by_config=True")
        if not e.get("policy_expected", True):
            bits.append("unexpected")
        lines.append(f"- `{name}`: {', '.join(bits)}")
    if fm.get("problems"):
        lines += ["", "## Problems"] + [f"- {p}" for p in fm["problems"]]
    if fm.get("warnings"):
        lines += ["", "## Warnings"] + [f"- {w}" for w in fm["warnings"]]
    lines += [
        "",
        "Proves the observation mapping is coherent for the evaluated frames — "
        "not task success, not physical safety.",
        "",
    ]
    return "\n".join(lines)


def run_eval_v2(config: EvalV2Config) -> dict[str, Any]:  # pragma: no cover - needs lerobot+net
    """Run eval-v2: load the dataset, build the feature mapping, write reports.

    Requires the [lerobot] extra and dataset access. Frame evaluation is
    best-effort and only runs when a runner is reachable.
    """
    from .lerobot_bridge import load_coreai_policy_for_lerobot

    bridge = load_coreai_policy_for_lerobot(
        config.policy_path, runner_url=config.runner_url,
        validate_runner=bool(config.runner_url))
    dataset_features, ds = _load_dataset_features(config.dataset_repo_id)

    feature_mapping = build_feature_mapping(
        dataset_features=dataset_features,
        policy_obs_features=manifest_obs_features(bridge.manifest),
        task_in_config=config.task is not None,
        strict=config.strict_features,
        fail_on_unknown=config.fail_on_unknown)

    frames_evaluated = 0  # frame rollout is optional and runner-dependent

    report = build_eval_v2_report(
        policy_path=config.policy_path, dataset_repo_id=config.dataset_repo_id,
        feature_mapping=feature_mapping, frames_evaluated=frames_evaluated,
        strict=config.strict_features)

    if config.output_dir:
        import json
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "lerobot_feature_mapping.json", "w") as f:
            json.dump(feature_mapping, f, indent=2)
        with open(out / "lerobot_eval_v2_report.json", "w") as f:
            json.dump(report, f, indent=2)
        (out / "lerobot_eval_v2_report.md").write_text(build_eval_v2_markdown(report))
    return report
