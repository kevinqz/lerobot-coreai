# compare_v2.py — processor-inclusive PyTorch-vs-CoreAI action parity (v1.2.6, hardened v1.2.7).
#
# compare-v2 runs BOTH sides through their declared processing and compares the
# FINAL action in the same unit. v1.2.7 hardens the EVIDENCE INTEGRITY:
#   - parity is a claim only when explicit numeric tolerances (gates) pass;
#   - structural (nested) shapes must match, not just flattened length;
#   - the compare target (next_action vs action_chunk) is explicit, never mixed.
# Still software-only: no robot/sim/real egress; no task-success/physical-safety
# claim. compare-v2 is EXPERIMENTAL evidence until manifest-v1 processor contracts
# and live temporal fixtures land (v1.2.8+).

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError

COMPARE_V2_SCHEMA_VERSION = "lerobot-coreai.compare_v2.v0"

VALID_TARGETS = ("next_action", "action_chunk")


def _structure(action: Any) -> Any:
    """Return the nested shape of an action (rectangular lengths per level)."""
    if isinstance(action, (list, tuple)):
        if not action:
            return (0,)
        inner = [_structure(e) for e in action]
        first = inner[0]
        # Rectangular if all inner structures agree.
        if all(s == first for s in inner):
            return (len(action),) + (first if isinstance(first, tuple) else ())
        return ("ragged", len(action))
    return ()  # scalar


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
    """Compute parity metrics; require STRUCTURAL shape match, not flat length."""
    frames = min(len(source_actions), len(coreai_actions))
    base = {"frames_compared": frames, "shape_match": False, "finite": False,
            "mae": None, "max_abs_error": None, "cosine_similarity": None,
            "relative_mae": None}
    if len(source_actions) != len(coreai_actions):
        return {**base, "detail": f"frame count differs: "
                f"{len(source_actions)} vs {len(coreai_actions)}"}
    if frames == 0:
        return {**base, "detail": "no frames compared"}

    s_all: list[float] = []
    c_all: list[float] = []
    for i in range(frames):
        if _structure(source_actions[i]) != _structure(coreai_actions[i]):
            return {**base, "detail": f"structural shape mismatch at frame {i}: "
                    f"{_structure(source_actions[i])} != {_structure(coreai_actions[i])}"}
        s_all.extend(_flatten(source_actions[i]))
        c_all.extend(_flatten(coreai_actions[i]))

    if not s_all:
        return {**base, "shape_match": True, "detail": "empty actions"}
    finite = all(math.isfinite(x) for x in s_all + c_all)
    if not finite:
        return {**base, "shape_match": True, "detail": "non-finite action values"}

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
    compare_target: str = "next_action"
    policy_revision: str | None = None
    dataset_revision: str | None = None
    # Numeric gates (None = disabled). Parity is claimed ONLY when configured
    # gates all pass — a huge finite error must never read as parity.
    max_mean_mae: float | None = None
    max_max_abs_error: float | None = None
    min_cosine_similarity: float | None = None
    max_relative_mae: float | None = None
    min_frames_compared: int = 1
    output_dir: Path | None = None


def evaluate_gates(metrics: dict[str, Any], config: CompareV2Config) -> dict[str, Any]:
    """Evaluate configured numeric gates against metrics."""
    gates: dict[str, Any] = {}

    def _le(name, value, threshold):
        if threshold is None or value is None:
            return
        gates[name] = {"value": value, "threshold": threshold,
                       "passed": value <= threshold}

    def _ge(name, value, threshold):
        if threshold is None or value is None:
            return
        gates[name] = {"value": value, "threshold": threshold,
                       "passed": value >= threshold}

    _le("mean_mae", metrics.get("mae"), config.max_mean_mae)
    _le("max_abs_error", metrics.get("max_abs_error"), config.max_max_abs_error)
    _ge("min_cosine_similarity", metrics.get("cosine_similarity"),
        config.min_cosine_similarity)
    _le("max_relative_mae", metrics.get("relative_mae"), config.max_relative_mae)
    return gates


# --- Mockable stage helpers (lerobot/runner gated) ---

def _load_frames(dataset_repo_id: str, max_frames: int, revision: str | None = None) -> list[Any]:  # pragma: no cover - needs lerobot
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = LeRobotDataset(dataset_repo_id, revision=revision) if revision else \
        LeRobotDataset(dataset_repo_id)
    return [ds[i] for i in range(min(max_frames, len(ds)))]


def _source_action(bundle, item, target: str) -> Any:  # pragma: no cover - needs lerobot
    pre = bundle.preprocessor(item)
    if target == "action_chunk":
        out = bundle.policy.predict_action_chunk(pre)
    else:
        out = bundle.policy.select_action(pre)
    return bundle.postprocessor(out)


def _coreai_action(coreai_policy, item, contract, target: str) -> Any:  # pragma: no cover - needs runner
    # target chooses the semantically-matching CoreAI method — never mix a
    # per-timestep action against a full chunk.
    if target == "action_chunk":
        return coreai_policy.predict_action_chunk(item)
    return coreai_policy.select_next_action(item)


def run_compare_v2(config: CompareV2Config) -> dict[str, Any]:
    """Run the processor-inclusive compare. Parity requires gates to pass."""
    if config.compare_target not in VALID_TARGETS:
        raise CoreAIPolicyError(
            f"--compare-target must be one of {VALID_TARGETS}, got "
            f"{config.compare_target!r}.")

    from .policy import CoreAIPolicy
    from .processor_contract import (
        build_processor_contract_report, parse_processor_contract_from_manifest,
    )
    from .source_loader_v2 import (
        SourceLoaderError, build_source_load_report, load_source_policy,
    )

    load_error = None
    bundle = None
    try:
        bundle = load_source_policy(config.torch_policy_path, config.dataset_repo_id,
                                    policy_revision=config.policy_revision,
                                    dataset_revision=config.dataset_revision)
    except SourceLoaderError as e:
        load_error = e
    source_report = build_source_load_report(
        config.torch_policy_path, config.dataset_repo_id, bundle=bundle, error=load_error)

    coreai = CoreAIPolicy.from_pretrained(config.coreai_policy_path,
                                          runner_url=config.runner_url)
    contract = parse_processor_contract_from_manifest(
        coreai.manifest, strict=config.strict_processors)
    contract_report = build_processor_contract_report(
        contract, strict=config.strict_processors)

    metrics: dict[str, Any] = {"frames_compared": 0, "shape_match": False,
                               "finite": False}
    per_frame: list[dict[str, Any]] = []
    if bundle is not None:
        frames = _load_frames(config.dataset_repo_id, config.max_frames,
                              config.dataset_revision)
        src_actions, coreai_actions = [], []
        for i, item in enumerate(frames):
            s = _source_action(bundle, item, config.compare_target)
            c = _coreai_action(coreai, item, contract, config.compare_target)
            src_actions.append(s)
            coreai_actions.append(c)
            per_frame.append({
                "frame_index": i, "compare_target": config.compare_target,
                "source_shape": list(_structure(s)), "coreai_shape": list(_structure(c)),
                "source_action": s, "coreai_action": c,
            })
        metrics = compute_compare_metrics(src_actions, coreai_actions)

    gates = evaluate_gates(metrics, config)
    gates_configured = bool(gates)
    gates_passed = all(g["passed"] for g in gates.values()) if gates_configured else False
    structural_ok = (
        metrics.get("shape_match") is True and metrics.get("finite") is True
        and metrics.get("frames_compared", 0) >= config.min_frames_compared)
    contract_ok = not contract.is_ambiguous()
    source_ok = bundle is not None and source_report.get("weights", {}).get("pretrained_path_bound")

    # rc0 "ran cleanly" — but PARITY is only claimed when gates are configured AND pass.
    ok = bool(structural_ok and contract_ok and source_ok
              and (gates_passed if gates_configured else True))
    parity_proven = bool(structural_ok and contract_ok and source_ok
                         and gates_configured and gates_passed)

    report = build_compare_v2_report(
        config, ok=ok, parity_proven=parity_proven, source_report=source_report,
        contract_report=contract_report, metrics=metrics, gates=gates,
        gates_configured=gates_configured)

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


def build_compare_v2_report(config: CompareV2Config, *, ok: bool, parity_proven: bool,
                            source_report: dict[str, Any],
                            contract_report: dict[str, Any], metrics: dict[str, Any],
                            gates: dict[str, Any], gates_configured: bool) -> dict[str, Any]:
    return {
        "schema_version": COMPARE_V2_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "experimental": True,
        "ok": ok,
        "compare_target": config.compare_target,
        "torch_policy_path": config.torch_policy_path,
        "coreai_policy_path": config.coreai_policy_path,
        "dataset_repo_id": config.dataset_repo_id,
        "strict_processors": config.strict_processors,
        "source_load": source_report,
        "processor_contract": contract_report,
        "metrics": metrics,
        "gates": gates,
        "gates_configured": gates_configured,
        "claims": {
            # Only true when numeric tolerances were configured AND met.
            "proves_action_parity_on_final_unit": parity_proven,
            "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }


def build_compare_v2_markdown(report: dict[str, Any]) -> str:
    m = report.get("metrics", {})
    lines = [
        "# Compare v2 — Processor-Inclusive Action Parity (EXPERIMENTAL)",
        "",
        f"- OK (ran + gated): {report.get('ok')}",
        f"- Parity proven: {report['claims']['proves_action_parity_on_final_unit']}"
        f" (gates configured: {report.get('gates_configured')})",
        f"- Compare target: {report.get('compare_target')}",
        f"- Torch policy: {report.get('torch_policy_path')}",
        f"- CoreAI policy: {report.get('coreai_policy_path')}",
        "",
        "## Metrics",
        f"- frames_compared: {m.get('frames_compared')}",
        f"- shape_match: {m.get('shape_match')}  finite: {m.get('finite')}",
        f"- mae: {m.get('mae')}  max_abs_error: {m.get('max_abs_error')}",
        f"- cosine_similarity: {m.get('cosine_similarity')}  relative_mae: {m.get('relative_mae')}",
    ]
    if report.get("gates"):
        lines += ["", "## Gates"]
        for name, g in report["gates"].items():
            mark = "✅" if g["passed"] else "❌"
            lines.append(f"- {mark} {name}: {g['value']} (threshold {g['threshold']})")
    lines += [
        "",
        "**Experimental.** Parity is claimed only when numeric tolerance gates are "
        "configured and pass. Proves neither task success nor physical safety.",
        "",
    ]
    return "\n".join(lines)
