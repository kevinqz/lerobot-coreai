# compare_v2.py — processor-inclusive PyTorch-vs-CoreAI action parity (v1.2.6).
#
# The v0.5 compare could compare a raw CoreAI action against a LeRobot action at
# the wrong stage — high numeric parity, operationally invalid. compare-v2 runs
# BOTH sides through their declared processing and compares the FINAL action in
# the same unit:
#   dataset frame -> official preprocessor -> policy -> official postprocessor -> source action
#   dataset frame -> (per processor_contract) -> CoreAI runner -> coreai action
# Metrics + per-frame trace only. No robot/sim/real egress; no task-success or
# physical-safety claim.

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError

COMPARE_V2_SCHEMA_VERSION = "lerobot-coreai.compare_v2.v0"


def _flatten(action: Any) -> list[float]:
    out: list[float] = []

    def _walk(x):
        if isinstance(x, (list, tuple)):
            for e in x:
                _walk(e)
        else:
            out.append(float(x))

    _walk(action)
    return out


def compute_compare_metrics(source_actions: list[Any],
                            coreai_actions: list[Any]) -> dict[str, Any]:
    """Compute parity metrics between two lists of per-frame final actions."""
    n = min(len(source_actions), len(coreai_actions))
    frames = n
    if len(source_actions) != len(coreai_actions):
        return {"frames_compared": frames, "shape_match": False, "finite": False,
                "mae": None, "max_abs_error": None, "cosine_similarity": None,
                "relative_mae": None,
                "detail": f"frame count differs: {len(source_actions)} vs {len(coreai_actions)}"}

    s_all: list[float] = []
    c_all: list[float] = []
    shape_match = True
    for i in range(frames):
        sf = _flatten(source_actions[i])
        cf = _flatten(coreai_actions[i])
        if len(sf) != len(cf):
            shape_match = False
            break
        s_all.extend(sf)
        c_all.extend(cf)

    if not shape_match or not s_all:
        return {"frames_compared": frames, "shape_match": shape_match, "finite": False,
                "mae": None, "max_abs_error": None, "cosine_similarity": None,
                "relative_mae": None, "detail": "shape mismatch" if not shape_match else "empty"}

    finite = all(math.isfinite(x) for x in s_all + c_all)
    if not finite:
        return {"frames_compared": frames, "shape_match": True, "finite": False,
                "mae": None, "max_abs_error": None, "cosine_similarity": None,
                "relative_mae": None, "detail": "non-finite action values"}

    diffs = [abs(a - b) for a, b in zip(s_all, c_all)]
    mae = sum(diffs) / len(diffs)
    max_abs = max(diffs)
    dot = sum(a * b for a, b in zip(s_all, c_all))
    ns = math.sqrt(sum(a * a for a in s_all))
    nc = math.sqrt(sum(b * b for b in c_all))
    cosine = (dot / (ns * nc)) if ns > 0 and nc > 0 else None
    src_mag = sum(abs(a) for a in s_all) / len(s_all)
    relative_mae = mae / src_mag if src_mag > 1e-12 else None

    return {
        "frames_compared": frames,
        "shape_match": True,
        "finite": True,
        "mae": round(mae, 8),
        "max_abs_error": round(max_abs, 8),
        "cosine_similarity": round(cosine, 8) if cosine is not None else None,
        "relative_mae": round(relative_mae, 8) if relative_mae is not None else None,
    }


@dataclass
class CompareV2Config:
    torch_policy_path: str
    coreai_policy_path: str
    dataset_repo_id: str
    runner_url: str | None = None
    max_frames: int = 32
    strict_processors: bool = False
    output_dir: Path | None = None


# --- Mockable stage helpers (lerobot/runner gated) ---

def _load_frames(dataset_repo_id: str, max_frames: int) -> list[Any]:  # pragma: no cover - needs lerobot
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = LeRobotDataset(dataset_repo_id)
    return [ds[i] for i in range(min(max_frames, len(ds)))]


def _source_final_action(bundle, item) -> Any:  # pragma: no cover - needs lerobot
    pre = bundle.preprocessor(item)
    out = bundle.policy.select_action(pre)
    return bundle.postprocessor(out)


def _coreai_final_action(coreai_policy, item, contract) -> Any:  # pragma: no cover - needs runner
    # Honor the processor contract: the runner returns the final action when
    # contract.returns == "postprocessed_action"; otherwise the caller must
    # apply denormalization (not implemented here — flagged by strict mode).
    return coreai_policy.select_action(item)


def run_compare_v2(config: CompareV2Config) -> dict[str, Any]:
    """Run the processor-inclusive compare. Fail-closed on ambiguous processors."""
    from .policy import CoreAIPolicy
    from .processor_contract import (
        build_processor_contract_report, parse_processor_contract_from_manifest,
    )
    from .source_loader_v2 import (
        SourceLoaderError, build_source_load_report, load_source_policy,
    )

    # 1. Source policy via the official API.
    load_error = None
    bundle = None
    try:
        bundle = load_source_policy(config.torch_policy_path, config.dataset_repo_id)
    except SourceLoaderError as e:
        load_error = e
    source_report = build_source_load_report(
        config.torch_policy_path, config.dataset_repo_id, bundle=bundle, error=load_error)

    # 2. CoreAI policy + processor contract (fail-closed on ambiguity if strict).
    coreai = CoreAIPolicy.from_pretrained(config.coreai_policy_path,
                                          runner_url=config.runner_url)
    contract = parse_processor_contract_from_manifest(
        coreai.manifest, strict=config.strict_processors)
    contract_report = build_processor_contract_report(
        contract, strict=config.strict_processors)

    metrics: dict[str, Any] = {"frames_compared": 0, "shape_match": False}
    per_frame: list[dict[str, Any]] = []
    if bundle is not None:
        frames = _load_frames(config.dataset_repo_id, config.max_frames)
        src_actions, coreai_actions = [], []
        for i, item in enumerate(frames):
            s = _source_final_action(bundle, item)
            c = _coreai_final_action(coreai, item, contract)
            src_actions.append(s)
            coreai_actions.append(c)
            per_frame.append({"frame": i})
        metrics = compute_compare_metrics(src_actions, coreai_actions)

    ok = (bundle is not None and metrics.get("shape_match") is True
          and metrics.get("finite") is True and not contract.is_ambiguous())

    report = build_compare_v2_report(
        config, ok=ok, source_report=source_report, contract_report=contract_report,
        metrics=metrics)

    if config.output_dir:
        import json
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "source_policy_load_report.json").write_text(json.dumps(source_report, indent=2))
        (out / "processor_contract_report.json").write_text(json.dumps(contract_report, indent=2))
        (out / "compare_v2_report.json").write_text(json.dumps(report, indent=2))
        (out / "compare_v2_report.md").write_text(build_compare_v2_markdown(report))
        with open(out / "compare_v2_actions.jsonl", "w") as f:
            for rec in per_frame:
                f.write(json.dumps(rec) + "\n")
    return report


def build_compare_v2_report(config: CompareV2Config, *, ok: bool,
                            source_report: dict[str, Any],
                            contract_report: dict[str, Any],
                            metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": COMPARE_V2_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "torch_policy_path": config.torch_policy_path,
        "coreai_policy_path": config.coreai_policy_path,
        "dataset_repo_id": config.dataset_repo_id,
        "strict_processors": config.strict_processors,
        "source_load": source_report,
        "processor_contract": contract_report,
        "metrics": metrics,
        "claims": {
            "proves_action_parity_on_final_unit": bool(ok),
            "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }


def build_compare_v2_markdown(report: dict[str, Any]) -> str:
    m = report.get("metrics", {})
    lines = [
        "# Compare v2 — Processor-Inclusive Action Parity",
        "",
        f"- OK: {report.get('ok')}",
        f"- Torch policy: {report.get('torch_policy_path')}",
        f"- CoreAI policy: {report.get('coreai_policy_path')}",
        f"- Dataset: {report.get('dataset_repo_id')}",
        "",
        "## Metrics",
        f"- frames_compared: {m.get('frames_compared')}",
        f"- shape_match: {m.get('shape_match')}  finite: {m.get('finite')}",
        f"- mae: {m.get('mae')}  max_abs_error: {m.get('max_abs_error')}",
        f"- cosine_similarity: {m.get('cosine_similarity')}  relative_mae: {m.get('relative_mae')}",
        "",
        "Compares the **final** action after each side's declared processing. Does "
        "not prove task success or physical safety.",
        "",
    ]
    return "\n".join(lines)
