# eval.py — LeRobotDataset replay/eval for CoreAI-backed policies (v0.4).
#
# Reads dataset frames, calls CoreAI runner via predict_action, validates actions,
# writes actions.jsonl + eval_trace.jsonl + eval_report.json.
# No robot connection. No motor commands.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .dataset import LeRobotDatasetEvalConfig, dataset_item_to_observation_batch, load_lerobot_dataset
from .errors import CoreAIPolicyError
from .policy import CoreAIPolicy
from .trace import TraceWriter
from .validation import validate_robot_type


@dataclass
class EvalConfig:
    policy_path: str
    dataset_repo_id: str
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    output_dir: Path = Path("runs/eval")
    robot_type: str | None = None
    max_frames: int = 32
    start_index: int = 0
    stride: int = 1
    episodes: list[int] | None = None
    dataset_root: Path | None = None
    dataset_revision: str | None = None
    download_videos: bool = True
    video_backend: str | None = None
    strict_observation_keys: bool = False
    fail_fast: bool = False
    overwrite: bool = False


@dataclass
class EvalResult:
    ok: bool
    output_dir: Path
    report_path: Path
    trace_path: Path
    actions_path: Path
    report: dict[str, Any] = field(default_factory=dict)


def run_lerobot_dataset_eval(config: EvalConfig) -> EvalResult:
    """Run LeRobotDataset eval: read frames, call CoreAI runner, generate actions, write report.

    Flow:
    1. Prepare output_dir
    2. Load CoreAIPolicy (with runner validation)
    3. Validate robot type
    4. Load LeRobotDataset
    5. Select frame indices (start_index, stride, max_frames)
    6. For each frame: extract observation → predict_action → validate → write actions.jsonl
    7. Generate eval_report.json
    8. Guarantee no-actuation invariants

    Never connects to a robot. Never sends motor commands.
    """
    output_dir = Path(config.output_dir)
    actions_path = output_dir / "actions.jsonl"
    trace_path = output_dir / "eval_trace.jsonl"
    report_path = output_dir / "eval_report.json"

    # Prepare output dir.
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Output directory not empty: {output_dir}. Use --overwrite to replace."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceWriter(trace_path)
    trace.write("eval.started", {"policy": config.policy_path, "dataset": config.dataset_repo_id})

    # Metrics accumulators.
    frames_requested = 0
    frames_processed = 0
    actions_generated = 0
    actions_failed = 0
    shape_errors = 0
    nan_errors = 0
    inf_errors = 0
    runner_errors = 0
    total_times: list[float] = []
    errors_list: list[dict[str, Any]] = []

    stage = "init"
    try:
        # Load policy.
        stage = "policy.load"
        trace.write("policy.loading")
        policy = CoreAIPolicy.from_pretrained(
            config.policy_path,
            runner_url=config.runner_url,
            validate_runner=True,
            return_metadata=True,
            strict_observation_keys=config.strict_observation_keys,
        )
        trace.write("policy.loaded", {"policy_type": policy.policy_type})
        trace.write("runner.checked")

        # Validate robot type.
        stage = "robot_type.validation"
        validate_robot_type(config.robot_type, policy.manifest)

        # Load dataset.
        stage = "dataset.load"
        trace.write("dataset.loading", {"repo_id": config.dataset_repo_id})
        ds_config = LeRobotDatasetEvalConfig(
            dataset_repo_id=config.dataset_repo_id,
            root=config.dataset_root,
            revision=config.dataset_revision,
            episodes=config.episodes,
            download_videos=config.download_videos,
            video_backend=config.video_backend,
        )
        dataset = load_lerobot_dataset(ds_config)
        num_available = len(dataset)
        trace.write("dataset.loaded", {"num_frames": num_available})

        # Select frame indices.
        indices = list(range(
            config.start_index,
            min(config.start_index + config.max_frames * config.stride, num_available),
            config.stride,
        ))
        frames_requested = len(indices)

        # Open actions.jsonl.
        actions_file = open(actions_path, "w")

        for frame_idx, ds_idx in enumerate(indices):
            stage = f"frame.{frame_idx}"
            trace.write("frame.started", {"frame_index": frame_idx, "dataset_index": ds_idx})

            try:
                item = dataset[ds_idx]
                batch = dataset_item_to_observation_batch(item, policy.manifest)
                # Convert tensors/arrays/images to JSON-safe values before sending to runner.
                from .serialization import make_json_safe_observation
                batch = make_json_safe_observation(
                    batch, output_dir=output_dir, frame_index=frame_idx,
                )
                result = policy.predict_action(batch, return_metadata=True)
                action = result["action"]
                metadata = result.get("metadata", {})
                timing = metadata.get("timing", {})

                actions_generated += 1
                frames_processed += 1
                if timing.get("total_ms"):
                    total_times.append(timing["total_ms"])

                # Write actions.jsonl line.
                entry = {
                    "frame_index": frame_idx,
                    "dataset_index": ds_idx,
                    "action": action,
                    "action_shape": _safe_shape(action),
                    "timing": timing,
                    "ok": True,
                    "error": None,
                }
                actions_file.write(json.dumps(entry) + "\n")
                actions_file.flush()

                trace.write("frame.action_generated", {"frame_index": frame_idx})

            except Exception as e:
                actions_failed += 1
                err_type = type(e).__name__
                err_msg = str(e)

                if "shape" in err_msg.lower():
                    shape_errors += 1
                elif "nan" in err_msg.lower():
                    nan_errors += 1
                elif "inf" in err_msg.lower():
                    inf_errors += 1
                elif "runner" in err_type.lower():
                    runner_errors += 1

                errors_list.append({
                    "type": err_type,
                    "message": err_msg,
                    "stage": stage,
                    "frame_index": frame_idx,
                })

                # Write failure line.
                fail_entry = {
                    "frame_index": frame_idx,
                    "dataset_index": ds_idx,
                    "ok": False,
                    "error": {"type": err_type, "message": err_msg, "stage": stage},
                }
                actions_file.write(json.dumps(fail_entry) + "\n")
                actions_file.flush()

                trace.write("frame.failed", {"frame_index": frame_idx, "error": err_type})

                if config.fail_fast:
                    trace.write("eval.failed", {"reason": "fail_fast", "frame": frame_idx})
                    actions_file.close()
                    raise

        actions_file.close()

        # Build report.
        mean_ms = sum(total_times) / len(total_times) if total_times else None
        p95_ms = sorted(total_times)[int(len(total_times) * 0.95)] if len(total_times) > 0 else None

        report = {
            "schema_version": "lerobot-coreai.eval_report.v0",
            "lerobot_coreai_version": __version__,
            "ok": actions_failed == 0,
            "mode": "dataset_eval",
            "policy": {
                "path": config.policy_path,
                "repo_id": config.policy_path,
                "source_repo_id": policy.manifest.policy_source_repo_id,
                "type": policy.policy_type,
                "runtime": "coreai",
                "model_id": policy.manifest.model_id,
            },
            "dataset": {
                "repo_id": config.dataset_repo_id,
                "revision": config.dataset_revision,
                "episodes": config.episodes,
                "num_frames_available": num_available,
            },
            "runner": {
                "url": config.runner_url,
                "reachable": True,
                "supports_action": True,
            },
            "metrics": {
                "frames_requested": frames_requested,
                "frames_processed": frames_processed,
                "actions_generated": actions_generated,
                "actions_failed": actions_failed,
                "shape_errors": shape_errors,
                "nan_errors": nan_errors,
                "inf_errors": inf_errors,
                "runner_errors": runner_errors,
                "mean_total_ms": mean_ms,
                "p95_total_ms": p95_ms,
            },
            "safety": {
                "physical_actuation_possible": False,
                "motor_commands_available": False,
                "robot_connected": False,
                "actions_sent": 0,
            },
            "files": {
                "actions": "actions.jsonl",
                "trace": "eval_trace.jsonl",
                "report": "eval_report.json",
            },
            "errors": errors_list,
        }

        report_path.write_text(json.dumps(report, indent=2) + "\n")
        trace.write("eval.completed", {"ok": report["ok"]})
        trace.close()

        return EvalResult(
            ok=report["ok"],
            output_dir=output_dir,
            report_path=report_path,
            trace_path=trace_path,
            actions_path=actions_path,
            report=report,
        )

    except Exception as e:
        # Best-effort failure report.
        error_type = type(e).__name__
        error_msg = str(e)

        try:
            fail_report = {
                "schema_version": "lerobot-coreai.eval_report.v0",
                "lerobot_coreai_version": __version__,
                "ok": False,
                "mode": "dataset_eval",
                "policy": {"path": config.policy_path},
                "dataset": {"repo_id": config.dataset_repo_id},
                "runner": {"url": config.runner_url, "reachable": False},
                "metrics": {
                    "frames_requested": frames_requested,
                    "frames_processed": frames_processed,
                    "actions_generated": actions_generated,
                    "actions_failed": actions_failed,
                    "shape_errors": shape_errors,
                    "nan_errors": nan_errors,
                    "inf_errors": inf_errors,
                    "runner_errors": runner_errors,
                    "mean_total_ms": None,
                    "p95_total_ms": None,
                },
                "safety": {
                    "physical_actuation_possible": False,
                    "motor_commands_available": False,
                    "robot_connected": False,
                    "actions_sent": 0,
                },
                "errors": [{"type": error_type, "message": error_msg, "stage": stage}],
            }
            report_path.write_text(json.dumps(fail_report, indent=2) + "\n")
        except Exception:
            pass

        trace.write("eval.failed", {"error": error_type, "message": error_msg})
        trace.close()
        raise


def _safe_shape(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) > 0:
        inner = _safe_shape(value[0])
        return [len(value)] + (inner or [])
    return None
