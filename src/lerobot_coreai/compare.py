# compare.py — PyTorch vs CoreAI action parity on LeRobotDataset (v0.5).
#
# Given identical dataset frames, runs both policies and compares actions.
# No robot. No motor commands. No task success claims. Only numeric fidelity.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .dataset import LeRobotDatasetEvalConfig, dataset_item_to_observation_batch, load_lerobot_dataset
from .errors import ActionParityError, CoreAIPolicyError
from .lerobot_adapter import load_lerobot_policy, make_torch_policy_batch
from .metrics import cosine_similarity, max_absolute_error, mean_absolute_error, relative_mae
from .policy import CoreAIPolicy
from .serialization import make_json_safe_observation
from .trace import TraceWriter
from .validation import validate_robot_type


@dataclass
class CompareConfig:
    torch_policy_path: str
    coreai_policy_path: str
    dataset_repo_id: str
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    output_dir: Path = Path("runs/compare")
    robot_type: str | None = None
    torch_policy_type: str | None = None
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
    tolerance_cosine: float = 0.999
    tolerance_max_mae: float = 1e-4
    tolerance_mean_mae: float = 1e-5
    save_actions: bool = False
    reset_each_frame: bool = False


@dataclass
class CompareResult:
    ok: bool
    output_dir: Path
    report_path: Path
    trace_path: Path
    actions_path: Path
    report: dict[str, Any] = field(default_factory=dict)


def run_lerobot_policy_compare(config: CompareConfig) -> CompareResult:
    """Run PyTorch vs CoreAI action parity comparison on LeRobotDataset frames.

    Flow:
    1. Load CoreAIPolicy (with runner validation)
    2. Load source PyTorch LeRobot policy
    3. Reset both policies
    4. Load LeRobotDataset
    5. For each frame: run both policies → compute metrics → write actions.jsonl
    6. Aggregate metrics
    7. Write compare_report.json
    8. No-actuation invariants

    Never connects to a robot. Never sends motor commands.
    """
    output_dir = Path(config.output_dir)
    actions_path = output_dir / "compare_actions.jsonl"
    trace_path = output_dir / "compare_trace.jsonl"
    report_path = output_dir / "compare_report.json"

    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Output directory not empty: {output_dir}. Use --overwrite to replace."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceWriter(trace_path)
    trace.write("compare.started", {
        "torch_policy": config.torch_policy_path,
        "coreai_policy": config.coreai_policy_path,
    })

    # Metrics accumulators.
    frames_requested = 0
    frames_compared = 0
    frames_passed = 0
    frames_failed = 0
    shape_mismatches = 0
    runner_errors = 0
    torch_errors = 0
    comparison_errors = 0
    cosines: list[float] = []
    maes: list[float] = []
    max_maes: list[float] = []
    rel_maes: list[float] = []
    coreai_times: list[float] = []
    errors_list: list[dict[str, Any]] = []

    stage = "init"
    try:
        # Load CoreAI policy.
        stage = "coreai_policy.load"
        trace.write("coreai_policy.loading")
        coreai_policy = CoreAIPolicy.from_pretrained(
            config.coreai_policy_path,
            runner_url=config.runner_url,
            validate_runner=True,
            return_metadata=True,
            strict_observation_keys=config.strict_observation_keys,
        )
        trace.write("coreai_policy.loaded", {"policy_type": coreai_policy.policy_type})

        # Validate robot type.
        stage = "robot_type.validation"
        validate_robot_type(config.robot_type, coreai_policy.manifest)

        # Load PyTorch policy.
        stage = "torch_policy.load"
        trace.write("torch_policy.loading")
        torch_policy = load_lerobot_policy(
            config.torch_policy_path,
            policy_type=config.torch_policy_type or coreai_policy.policy_type,
        )
        trace.write("torch_policy.loaded")

        # Reset both policies.
        if hasattr(torch_policy, "reset"):
            torch_policy.reset()
        coreai_policy.reset()

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

        actions_file = open(actions_path, "w")
        actions_dir = output_dir / "actions" if config.save_actions else None
        if actions_dir:
            actions_dir.mkdir(exist_ok=True)

        for frame_idx, ds_idx in enumerate(indices):
            stage = f"frame.{frame_idx}"
            trace.write("frame.started", {"frame_index": frame_idx, "dataset_index": ds_idx})

            if config.reset_each_frame:
                if hasattr(torch_policy, "reset"):
                    torch_policy.reset()
                coreai_policy.reset()

            try:
                item = dataset[ds_idx]
                raw_batch = dataset_item_to_observation_batch(item, coreai_policy.manifest)
                json_batch = make_json_safe_observation(
                    raw_batch, output_dir=output_dir, frame_index=frame_idx,
                )
                torch_batch = make_torch_policy_batch(raw_batch)

                # PyTorch action.
                stage = f"frame.{frame_idx}.torch"
                torch_action = torch_policy.select_action(torch_batch)
                trace.write("frame.torch_action_generated", {"frame_index": frame_idx})

                # CoreAI action.
                stage = f"frame.{frame_idx}.coreai"
                coreai_result = coreai_policy.predict_action(json_batch, return_metadata=True)
                coreai_action = coreai_result["action"]
                coreai_timing = coreai_result.get("metadata", {}).get("timing", {})
                trace.write("frame.coreai_action_generated", {"frame_index": frame_idx})

                # Compare.
                stage = f"frame.{frame_idx}.compare"
                cos = cosine_similarity(torch_action, coreai_action)
                mae = mean_absolute_error(torch_action, coreai_action)
                max_mae = max_absolute_error(torch_action, coreai_action)
                rel = relative_mae(torch_action, coreai_action)

                cosines.append(cos)
                maes.append(mae)
                max_maes.append(max_mae)
                rel_maes.append(rel)
                if coreai_timing.get("total_ms"):
                    coreai_times.append(coreai_timing["total_ms"])

                passed = (
                    cos >= config.tolerance_cosine
                    and max_mae <= config.tolerance_max_mae
                    and mae <= config.tolerance_mean_mae
                )
                frames_compared += 1
                if passed:
                    frames_passed += 1
                else:
                    frames_failed += 1

                entry: dict[str, Any] = {
                    "frame_index": frame_idx,
                    "dataset_index": ds_idx,
                    "ok": True,
                    "passed": passed,
                    "metrics": {
                        "cosine_similarity": cos,
                        "mean_absolute_error": mae,
                        "max_absolute_error": max_mae,
                        "relative_mae": rel,
                    },
                    "timing": {"coreai_total_ms": coreai_timing.get("total_ms")},
                    "error": None,
                }

                if config.save_actions and actions_dir:
                    torch_path = actions_dir / f"frame_{frame_idx:06d}_torch.json"
                    coreai_path = actions_dir / f"frame_{frame_idx:06d}_coreai.json"
                    from .serialization import _tensor_to_list
                    torch_path.write_text(json.dumps(_tensor_to_list(torch_action), indent=2))
                    coreai_path.write_text(json.dumps(coreai_action, indent=2))
                    entry["torch_action_path"] = str(torch_path.relative_to(output_dir))
                    entry["coreai_action_path"] = str(coreai_path.relative_to(output_dir))

                actions_file.write(json.dumps(entry) + "\n")
                actions_file.flush()

                trace.write("frame.compared", {
                    "frame_index": frame_idx, "passed": passed,
                    "cosine": cos, "mae": mae,
                })

            except ActionParityError as e:
                frames_failed += 1
                shape_mismatches += 1
                comparison_errors += 1
                _record_frame_failure(actions_file, frame_idx, ds_idx, "ActionParityError", str(e), stage)
                trace.write("frame.failed", {"frame_index": frame_idx, "error": "ActionParityError"})
                errors_list.append({"type": "ActionParityError", "message": str(e), "stage": stage,
                                    "frame_index": frame_idx})
                if config.fail_fast:
                    raise

            except CoreAIPolicyError as e:
                frames_failed += 1
                runner_errors += 1
                _record_frame_failure(actions_file, frame_idx, ds_idx, type(e).__name__, str(e), stage)
                trace.write("frame.failed", {"frame_index": frame_idx, "error": type(e).__name__})
                errors_list.append({"type": type(e).__name__, "message": str(e), "stage": stage,
                                    "frame_index": frame_idx})
                if config.fail_fast:
                    raise

            except Exception as e:
                frames_failed += 1
                torch_errors += 1
                _record_frame_failure(actions_file, frame_idx, ds_idx, type(e).__name__, str(e), stage)
                trace.write("frame.failed", {"frame_index": frame_idx, "error": type(e).__name__})
                errors_list.append({"type": type(e).__name__, "message": str(e), "stage": stage,
                                    "frame_index": frame_idx})
                if config.fail_fast:
                    raise

        actions_file.close()

        # Aggregate.
        mean_cos = sum(cosines) / len(cosines) if cosines else None
        min_cos = min(cosines) if cosines else None
        mean_mae = sum(maes) / len(maes) if maes else None
        max_mae_agg = max(max_maes) if max_maes else None
        mean_rel = sum(rel_maes) / len(rel_maes) if rel_maes else None
        p95_ms = sorted(coreai_times)[int(len(coreai_times) * 0.95)] if coreai_times else None

        ok = frames_compared > 0 and frames_failed == 0
        proves_fidelity = ok and frames_compared >= 1

        report = {
            "schema_version": "lerobot-coreai.compare_report.v0",
            "lerobot_coreai_version": __version__,
            "ok": ok,
            "mode": "dataset_compare",
            "policy": {
                "torch": {"path": config.torch_policy_path, "type": config.torch_policy_type or coreai_policy.policy_type},
                "coreai": {
                    "path": config.coreai_policy_path,
                    "repo_id": config.coreai_policy_path,
                    "source_repo_id": coreai_policy.manifest.policy_source_repo_id,
                    "type": coreai_policy.policy_type,
                    "runtime": "coreai",
                    "model_id": coreai_policy.manifest.model_id,
                },
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
            "tolerances": {
                "cosine_similarity_min": config.tolerance_cosine,
                "max_absolute_error_max": config.tolerance_max_mae,
                "mean_absolute_error_max": config.tolerance_mean_mae,
            },
            "metrics": {
                "frames_requested": frames_requested,
                "frames_compared": frames_compared,
                "frames_passed": frames_passed,
                "frames_failed": frames_failed,
                "shape_mismatches": shape_mismatches,
                "runner_errors": runner_errors,
                "torch_errors": torch_errors,
                "comparison_errors": comparison_errors,
                "mean_cosine_similarity": mean_cos,
                "min_cosine_similarity": min_cos,
                "mean_absolute_error": mean_mae,
                "max_absolute_error": max_mae_agg,
                "mean_relative_mae": mean_rel,
                "p95_coreai_total_ms": p95_ms,
            },
            "claims": {
                "proves_numeric_action_fidelity": proves_fidelity,
                "proves_task_success": False,
                "proves_robot_safety": False,
            },
            "safety": {
                "physical_actuation_possible": False,
                "motor_commands_available": False,
                "robot_connected": False,
                "actions_sent": 0,
            },
            "files": {
                "actions": "compare_actions.jsonl",
                "trace": "compare_trace.jsonl",
                "report": "compare_report.json",
            },
            "errors": errors_list,
        }

        report_path.write_text(json.dumps(report, indent=2) + "\n")

        # Write manifest evaluation patch if passed.
        if ok:
            patch_path = output_dir / "manifest-evaluation-patch.json"
            patch = {
                "evaluation": {
                    "metric": "action_parity",
                    "status": "passed",
                    "n_obs": frames_compared,
                    "min_chunk_cosine": min_cos,
                    "max_action_mae": max_mae_agg,
                    "mean_action_mae": mean_mae,
                    "max_relative_action_mae": mean_rel,
                    "proves_numeric_fidelity": True,
                    "proves_task_success": False,
                    "proves_robot_safety": False,
                }
            }
            patch_path.write_text(json.dumps(patch, indent=2) + "\n")

        trace.write("compare.completed", {"ok": ok})
        trace.close()

        return CompareResult(
            ok=ok,
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
                "schema_version": "lerobot-coreai.compare_report.v0",
                "lerobot_coreai_version": __version__,
                "ok": False,
                "mode": "dataset_compare",
                "policy": {"torch": {"path": config.torch_policy_path}, "coreai": {"path": config.coreai_policy_path}},
                "dataset": {"repo_id": config.dataset_repo_id},
                "runner": {"url": config.runner_url, "reachable": False},
                "metrics": {"frames_requested": frames_requested, "frames_compared": frames_compared,
                            "frames_passed": frames_passed, "frames_failed": frames_failed},
                "claims": {"proves_numeric_action_fidelity": False, "proves_task_success": False, "proves_robot_safety": False},
                "safety": {"physical_actuation_possible": False, "motor_commands_available": False,
                           "robot_connected": False, "actions_sent": 0},
                "errors": [{"type": error_type, "message": error_msg, "stage": stage}],
            }
            report_path.write_text(json.dumps(fail_report, indent=2) + "\n")
        except Exception:
            pass

        trace.write("compare.failed", {"error": error_type, "message": error_msg})
        trace.close()
        raise


def _record_frame_failure(
    actions_file, frame_idx: int, ds_idx: int, error_type: str, error_msg: str, stage: str,
) -> None:
    entry = {
        "frame_index": frame_idx,
        "dataset_index": ds_idx,
        "ok": False,
        "passed": False,
        "error": {"type": error_type, "message": error_msg, "stage": stage},
    }
    actions_file.write(json.dumps(entry) + "\n")
    actions_file.flush()
