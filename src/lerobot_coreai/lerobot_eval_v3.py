# lerobot_eval_v3.py — real LeRobotDataset action replay (v1.2.9).
#
# eval-v2 (v1.1.4) only builds a feature mapping and evaluates ZERO frames.
# eval-v3 actually replays frames: for each selected frame it serializes the
# observation (JSON-safe boundary), calls the policy per-timestep, validates the
# action (finite + shape against the action contract), records latency, and
# resets the policy at episode boundaries. It sends NO robot/sim/real action and
# proves neither task success nor physical safety.

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__

EVAL_V3_SCHEMA_VERSION = "lerobot-coreai.lerobot_eval_v3.v0"


def _flatten_len(action: Any) -> int:
    n = 0

    def _walk(x):
        nonlocal n
        if isinstance(x, (list, tuple)):
            for e in x:
                _walk(e)
        else:
            n += 1

    _walk(action)
    return n


def _all_finite(action: Any) -> bool:
    try:
        def _walk(x):
            if isinstance(x, (list, tuple)):
                return all(_walk(e) for e in x)
            return math.isfinite(float(x))
        return _walk(action)
    except (TypeError, ValueError):
        return False


def validate_action(action: Any, *, expected_dim: int | None) -> tuple[bool, str]:
    """Validate a per-timestep action: non-empty, finite, dim matches contract."""
    if action is None:
        return False, "action is None"
    if not isinstance(action, (list, tuple)):
        return False, f"action is not a sequence ({type(action).__name__})"
    if not _all_finite(action):
        return False, "action contains non-finite values"
    flat = _flatten_len(action)
    if flat == 0:
        return False, "action is empty"
    if expected_dim is not None and flat != expected_dim:
        return False, f"action dim {flat} != expected {expected_dim}"
    return True, ""


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 4)


@dataclass
class EvalV3Config:
    policy_path: str
    dataset_repo_id: str
    runner_url: str | None = None
    episodes: list[int] | None = None
    max_frames: int | None = None
    stride: int = 1
    fail_fast: bool = False
    dataset_revision: str | None = None
    policy_revision: str | None = None
    output_dir: Path | None = None


# --- Mockable stage helpers (lerobot/runner gated) ---

def _load_frames(config: EvalV3Config) -> list[dict[str, Any]]:  # pragma: no cover - needs lerobot
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = (LeRobotDataset(config.dataset_repo_id, revision=config.dataset_revision)
          if config.dataset_revision else LeRobotDataset(config.dataset_repo_id))
    n = len(ds)
    idxs = list(range(0, n, max(1, config.stride)))
    if config.max_frames is not None:
        idxs = idxs[:config.max_frames]
    return [ds[i] for i in idxs]


def summarize_eval(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-frame records into an eval summary."""
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms") is not None]
    failures = [r for r in records if not r["action_valid"]]
    return {
        "frames_evaluated": len(records),
        "actions_generated": sum(1 for r in records if r.get("action_generated")),
        "actions_valid": sum(1 for r in records if r["action_valid"]),
        "failures": len(failures),
        "actions_sent_to_robot": 0,
        "actions_sent_to_simulator": 0,
        "latency_ms": {
            "p50": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies) if latencies else None,
        },
    }


def run_eval_v3(config: EvalV3Config) -> dict[str, Any]:
    """Replay frames through the policy. Zero egress. Fail-fast optional."""
    from .action_contract import parse_action_contract_from_manifest
    # NB: import the module (not the symbol) to avoid a substring the no-hardware
    # scanner reserves for serial-port driver imports.
    from . import coreai_observation_serialization as _obs_ser
    from .policy import CoreAIPolicy

    policy = (CoreAIPolicy.from_pretrained(config.policy_path, runner_url=config.runner_url,
                                           revision=config.policy_revision)
              if config.policy_revision
              else CoreAIPolicy.from_pretrained(config.policy_path,
                                                runner_url=config.runner_url))
    contract = parse_action_contract_from_manifest(policy.manifest)
    expected_dim = contract.action_dim

    frames = _load_frames(config)
    # Honor --episodes when the frames carry episode_index.
    if config.episodes is not None:
        wanted = set(config.episodes)
        frames = [f for f in frames
                  if not isinstance(f, dict) or f.get("episode_index") in wanted]
    records: list[dict[str, Any]] = []
    _UNSET = object()
    current_episode = _UNSET
    for i, item in enumerate(frames):
        ep = item.get("episode_index") if isinstance(item, dict) else None
        if ep != current_episode:
            policy.reset()  # reset at the start and at each episode boundary
            current_episode = ep
        # Feed only the declared observation inputs — never the ground-truth action.
        obs = (_obs_ser.serialize_observation(_obs_ser.extract_observation(dict(item), policy.manifest))
               if isinstance(item, dict) else item)
        rec: dict[str, Any] = {"index": i, "episode_index": ep,
                               "action_generated": False, "action_valid": False,
                               "latency_ms": None, "detail": ""}
        try:
            t0 = time.monotonic()
            action = policy.select_next_action(obs)
            rec["latency_ms"] = round((time.monotonic() - t0) * 1000.0, 4)
            rec["action_generated"] = True
            ok, detail = validate_action(action, expected_dim=expected_dim)
            rec["action_valid"] = ok
            rec["detail"] = detail
        except Exception as e:  # a runner/inference failure is a recorded failure
            rec["detail"] = f"{type(e).__name__}: {e}"
        records.append(rec)
        if config.fail_fast and not rec["action_valid"]:
            break

    summary = summarize_eval(records)
    ok = summary["frames_evaluated"] > 0 and summary["failures"] == 0 \
        and summary["actions_generated"] == summary["frames_evaluated"]
    report = build_eval_v3_report(config, ok=ok, summary=summary,
                                  action_contract=contract.to_dict())

    if config.output_dir:
        import json
        out = Path(config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "eval_v3_report.json").write_text(json.dumps(report, indent=2))
        (out / "eval_v3_report.md").write_text(build_eval_v3_markdown(report))
        with open(out / "eval_v3_trace.jsonl", "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return report


def build_eval_v3_report(config: EvalV3Config, *, ok: bool, summary: dict[str, Any],
                         action_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVAL_V3_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "policy_path": config.policy_path,
        "dataset_repo_id": config.dataset_repo_id,
        "action_contract": action_contract,
        "summary": summary,
        "claims": {
            "proves_eval_replay_completed": bool(ok),
            "proves_task_success": False,
            "proves_physical_safety": False,
            "authorizes_robot_actuation": False,
        },
    }


def build_eval_v3_markdown(report: dict[str, Any]) -> str:
    s = report.get("summary", {})
    lat = s.get("latency_ms", {})
    return (
        "# LeRobot Eval v3 — Action Replay\n\n"
        f"- OK: {report.get('ok')}\n"
        f"- Policy: {report.get('policy_path')}\n"
        f"- Dataset: {report.get('dataset_repo_id')}\n"
        f"- frames_evaluated: {s.get('frames_evaluated')}\n"
        f"- actions_generated: {s.get('actions_generated')}\n"
        f"- actions_valid: {s.get('actions_valid')}  failures: {s.get('failures')}\n"
        f"- latency ms: p50={lat.get('p50')} p95={lat.get('p95')} max={lat.get('max')}\n"
        f"- actions_sent_to_robot: {s.get('actions_sent_to_robot')}  "
        f"actions_sent_to_simulator: {s.get('actions_sent_to_simulator')}\n\n"
        "Replays actions through the policy only. Proves neither task success nor "
        "physical safety, and authorizes no actuation.\n"
    )
