# export.py — export/verify/package orchestration pipeline (v0.6).
#
# LeRobot PyTorch policy → coreai-fabric export → manifest → verify → publish folder.
# No robot. No motor commands. No task success claims.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError
from .trace import TraceWriter


@dataclass
class ExportConfig:
    torch_policy_path: str
    output_dir: Path = Path("runs/export")
    policy_type: str | None = None
    robot_type: str | None = None
    dataset_repo_id: str | None = None
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    model_id: str | None = None
    output_repo_id: str | None = None
    artifact_name: str | None = None
    fabric_config: Path | None = None
    fabric_profile: str | None = None
    fabric_target: str = "coreai"
    skip_fabric: bool = False
    existing_artifact: Path | None = None
    verify_runner: bool = False
    dry_run_fixture: Path | None = None
    eval_max_frames: int = 0
    compare_max_frames: int = 0
    compare_tolerance_cosine: float = 0.999
    compare_tolerance_max_mae: float = 1e-4
    compare_tolerance_mean_mae: float = 1e-5
    publish_ready: bool = False
    overwrite: bool = False
    fail_fast: bool = False


@dataclass
class ExportResult:
    ok: bool
    output_dir: Path
    report_path: Path
    trace_path: Path
    manifest_path: Path | None = None
    artifact_path: Path | None = None
    report: dict[str, Any] = field(default_factory=dict)


def run_coreai_export_pipeline(config: ExportConfig) -> ExportResult:
    """Run the full export → verify → package pipeline.

    Flow:
    1. Prepare output_dir
    2. Export via fabric (or use existing artifact)
    3. Generate/load manifest
    4. Optional: runner verify, dry_run, eval, compare
    5. Build export_report.json
    6. Optional: publish folder
    7. Safety invariants always preserved

    Never connects to a robot. Never sends motor commands.
    """
    output_dir = Path(config.output_dir)
    report_path = output_dir / "export_report.json"
    trace_path = output_dir / "export_trace.jsonl"
    manifest_path: Path | None = None
    artifact_path: Path | None = None

    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Output directory not empty: {output_dir}. Use --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceWriter(trace_path)
    trace.write("export.started", {"torch_policy": config.torch_policy_path})

    # Verification state.
    manifest_valid = False
    runner_checked = False
    runner_ok = False
    dry_run_result: dict[str, Any] | None = None
    eval_result: dict[str, Any] | None = None
    compare_result: dict[str, Any] | None = None
    proves_numeric_fidelity = False
    errors_list: list[dict[str, Any]] = []
    stage = "init"

    try:
        # Step 1: Export or discover artifact.
        if config.skip_fabric:
            stage = "artifact.discovery"
            if not config.existing_artifact:
                raise CoreAIPolicyError(
                    "--skip-fabric requires --existing-artifact to point to an existing .aimodel bundle."
                )
            artifact_path = Path(config.existing_artifact)
            if not artifact_path.exists():
                raise CoreAIPolicyError(f"Existing artifact not found: {artifact_path}")
            trace.write("artifact.discovered", {"path": str(artifact_path)})
        else:
            stage = "fabric.export"
            trace.write("fabric.export.started")
            from .fabric_adapter import FabricExportConfig, run_fabric_export
            fabric_config = FabricExportConfig(
                torch_policy_path=config.torch_policy_path,
                output_dir=output_dir,
                policy_type=config.policy_type,
                robot_type=config.robot_type,
                model_id=config.model_id,
                output_repo_id=config.output_repo_id,
                fabric_config=config.fabric_config,
                fabric_profile=config.fabric_profile,
                fabric_target=config.fabric_target,
                artifact_name=config.artifact_name,
            )
            fabric_result = run_fabric_export(fabric_config)
            artifact_path = fabric_result.artifact_path
            manifest_path = fabric_result.manifest_path
            trace.write("fabric.export.completed", {
                "artifact": str(artifact_path) if artifact_path else None,
                "model_id": fabric_result.model_id,
            })

        # Step 2: Manifest.
        stage = "manifest.generate"
        if manifest_path and manifest_path.exists():
            from .manifest import LeRobotCoreAIManifest
            manifest_data = json.loads(manifest_path.read_text())
            LeRobotCoreAIManifest.from_dict(manifest_data)  # validates
            manifest_valid = True
            trace.write("manifest.validated", {"path": str(manifest_path)})
        else:
            # Generate a minimal manifest.
            manifest_path = output_dir / "lerobot-coreai.json"
            model_id = config.model_id or config.torch_policy_path.split("/")[-1].lower()
            manifest_data = _build_minimal_manifest(
                torch_policy_path=config.torch_policy_path,
                model_id=model_id,
                policy_type=config.policy_type or "unknown",
                robot_type=config.robot_type or "unknown",
                output_repo_id=config.output_repo_id,
            )
            # Validate generated manifest against schema.
            from .manifest import LeRobotCoreAIManifest
            LeRobotCoreAIManifest.from_dict(manifest_data)
            manifest_path.write_text(json.dumps(manifest_data, indent=2) + "\n")
            manifest_valid = True
            trace.write("manifest.generated", {"path": str(manifest_path)})

        # Step 3: Optional verifications.
        if config.verify_runner:
            stage = "runner.verify"
            from .runner import RunnerClient
            rc = RunnerClient(config.runner_url)
            try:
                rc.health()
                rc.supports_action()
                runner_checked = True
                runner_ok = True
                trace.write("runner.checked", {"ok": True})
            finally:
                rc.close()

        if config.dry_run_fixture:
            stage = "dry_run"
            trace.write("dry_run.started")
            from .rollout import DryRunRolloutConfig, run_dry_run_rollout
            dry_config = DryRunRolloutConfig(
                policy_path=config.output_repo_id or config.torch_policy_path,
                robot_type=config.robot_type,
                fixture_path=Path(config.dry_run_fixture),
                runner_url=config.runner_url,
                output_dir=output_dir / "dry_run",
                overwrite=True,
            )
            try:
                dr = run_dry_run_rollout(dry_config)
                dry_run_result = {"ran": True, "ok": dr.ok, "report": "dry_run/rollout_report.json"}
                trace.write("dry_run.completed", {"ok": dr.ok})
            except Exception as e:
                dry_run_result = {"ran": True, "ok": False, "error": str(e)}
                errors_list.append({"type": type(e).__name__, "message": str(e), "stage": "dry_run"})
                if config.fail_fast:
                    raise

        if config.eval_max_frames > 0 and config.dataset_repo_id:
            stage = "eval"
            trace.write("eval.started")
            from .eval import EvalConfig, run_lerobot_dataset_eval
            eval_config = EvalConfig(
                policy_path=config.output_repo_id or config.torch_policy_path,
                dataset_repo_id=config.dataset_repo_id,
                runner_url=config.runner_url,
                output_dir=output_dir / "eval",
                robot_type=config.robot_type,
                max_frames=config.eval_max_frames,
                overwrite=True,
                fail_fast=config.fail_fast,
            )
            try:
                er = run_lerobot_dataset_eval(eval_config)
                eval_result = {"ran": True, "ok": er.ok, "frames_processed": er.report.get("metrics", {}).get("frames_processed", 0), "report": "eval/eval_report.json"}
                trace.write("eval.completed", {"ok": er.ok})
            except Exception as e:
                eval_result = {"ran": True, "ok": False, "error": str(e)}
                errors_list.append({"type": type(e).__name__, "message": str(e), "stage": "eval"})
                if config.fail_fast:
                    raise

        if config.compare_max_frames > 0 and config.dataset_repo_id:
            stage = "compare"
            trace.write("compare.started")
            from .compare import CompareConfig, run_lerobot_policy_compare
            compare_config = CompareConfig(
                torch_policy_path=config.torch_policy_path,
                coreai_policy_path=config.output_repo_id or config.torch_policy_path,
                dataset_repo_id=config.dataset_repo_id,
                runner_url=config.runner_url,
                output_dir=output_dir / "compare",
                robot_type=config.robot_type,
                torch_policy_type=config.policy_type,
                max_frames=config.compare_max_frames,
                overwrite=True,
                tolerance_cosine=config.compare_tolerance_cosine,
                tolerance_max_mae=config.compare_tolerance_max_mae,
                tolerance_mean_mae=config.compare_tolerance_mean_mae,
                fail_fast=config.fail_fast,
            )
            try:
                cr = run_lerobot_policy_compare(compare_config)
                proves_numeric_fidelity = cr.report.get("claims", {}).get("proves_numeric_action_fidelity", False)
                compare_result = {"ran": True, "ok": cr.ok, "frames_compared": cr.report.get("metrics", {}).get("frames_compared", 0), "proves_numeric_action_fidelity": proves_numeric_fidelity, "report": "compare/compare_report.json"}
                trace.write("compare.completed", {"ok": cr.ok, "fidelity": proves_numeric_fidelity})
            except Exception as e:
                compare_result = {"ran": True, "ok": False, "error": str(e)}
                errors_list.append({"type": type(e).__name__, "message": str(e), "stage": "compare"})
                if config.fail_fast:
                    raise

        # Determine overall ok.
        ok = manifest_valid
        if dry_run_result and not dry_run_result.get("ok"):
            ok = False
        if eval_result and not eval_result.get("ok"):
            ok = False
        if compare_result and not compare_result.get("ok"):
            ok = False
        ok = ok and len(errors_list) == 0

        # Build report.
        report = {
            "schema_version": "lerobot-coreai.export_report.v0",
            "lerobot_coreai_version": __version__,
            "ok": ok,
            "mode": "export_verify_package",
            "source": {
                "torch_policy_path": config.torch_policy_path,
                "policy_type": config.policy_type,
                "robot_type": config.robot_type,
            },
            "artifact": {
                "format": "aimodel",
                "path": str(artifact_path.relative_to(output_dir)) if artifact_path and output_dir in artifact_path.parents else str(artifact_path) if artifact_path else None,
                "model_id": config.model_id,
                "manifest": str(manifest_path.name) if manifest_path else None,
            },
            "fabric": {
                "used": not config.skip_fabric,
                "status": "passed" if not config.skip_fabric else "skipped",
                "profile": config.fabric_profile,
                "target": config.fabric_target,
            },
            "verification": {
                "manifest_valid": manifest_valid,
                "runner_checked": runner_checked,
                "runner_ok": runner_ok,
                "dry_run": dry_run_result or {"ran": False},
                "eval": eval_result or {"ran": False},
                "compare": compare_result or {"ran": False},
            },
            "claims": {
                "proves_numeric_action_fidelity": proves_numeric_fidelity,
                "proves_task_success": False,
                "proves_robot_safety": False,
                "publish_ready": config.publish_ready and ok,
            },
            "safety": {
                "physical_actuation_possible": False,
                "motor_commands_available": False,
                "robot_connected": False,
                "actions_sent": 0,
            },
            "files": {
                "manifest": "lerobot-coreai.json" if manifest_path else None,
                "trace": "export_trace.jsonl",
                "report": "export_report.json",
                "publish_dir": "publish/" if config.publish_ready and ok else None,
            },
            "errors": errors_list,
        }

        report_path.write_text(json.dumps(report, indent=2) + "\n")

        # Step 4: Publish folder.
        if config.publish_ready and ok:
            stage = "publish"
            _build_publish_folder(output_dir, artifact_path, manifest_path, report)
            trace.write("publish.prepared")

        trace.write("export.completed", {"ok": ok})
        trace.close()

        return ExportResult(
            ok=ok,
            output_dir=output_dir,
            report_path=report_path,
            trace_path=trace_path,
            manifest_path=manifest_path,
            artifact_path=artifact_path,
            report=report,
        )

    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        try:
            fail_report = {
                "schema_version": "lerobot-coreai.export_report.v0",
                "lerobot_coreai_version": __version__,
                "ok": False,
                "mode": "export_verify_package",
                "source": {"torch_policy_path": config.torch_policy_path},
                "artifact": {},
                "fabric": {"used": not config.skip_fabric, "status": "failed"},
                "verification": {"manifest_valid": manifest_valid},
                "claims": {"proves_numeric_action_fidelity": False,
                           "proves_task_success": False, "proves_robot_safety": False,
                           "publish_ready": False},
                "safety": {"physical_actuation_possible": False, "motor_commands_available": False,
                           "robot_connected": False, "actions_sent": 0},
                "errors": [{"type": error_type, "message": error_msg, "stage": stage}],
            }
            report_path.write_text(json.dumps(fail_report, indent=2) + "\n")
        except Exception:
            pass

        trace.write("export.failed", {"error": error_type, "message": error_msg})
        trace.close()
        raise


def _build_minimal_manifest(
    *, torch_policy_path: str, model_id: str, policy_type: str,
    robot_type: str, output_repo_id: str | None,
) -> dict[str, Any]:
    robot_block: dict[str, Any] = {"type": robot_type}
    policy_block: dict[str, Any] = {
        "repo_id": output_repo_id or torch_policy_path,
        "source_repo_id": torch_policy_path,
        "type": policy_type,
    }
    return {
        "schema_version": "lerobot-coreai.v0",
        "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0"},
        "policy": policy_block,
        "robot": robot_block,
        "features": {"observation": {}, "action": {}},
        "normalization": {"format": "lerobot", "path": "norm_stats.json"},
        "coreai": {
            "artifact_format": "aimodel", "runner": "coreai-runner",
            "model_id": model_id, "graphs": [], "host_loop_required": False,
        },
        "evaluation": {
            "status": "not_run",
            "proves_numeric_fidelity": False,
            "proves_task_success": False, "proves_robot_safety": False,
        },
        "safety": {"default_mode": "dry_run", "real_actuation_requires_confirmation": True},
    }


def _build_publish_folder(
    output_dir: Path, artifact_path: Path | None,
    manifest_path: Path | None, report: dict[str, Any],
) -> None:
    import shutil as _shutil
    publish_dir = output_dir / "publish"
    publish_dir.mkdir(exist_ok=True)

    # Copy manifest.
    if manifest_path and manifest_path.exists():
        _shutil.copy2(manifest_path, publish_dir / "lerobot-coreai.json")

    # Copy artifact.
    if artifact_path and artifact_path.exists():
        _shutil.copytree(artifact_path, publish_dir / artifact_path.name, dirs_exist_ok=True)

    # Copy reports.
    reports_dir = publish_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    for src, dst_name in [
        (output_dir / "export_report.json", "export_report.json"),
        (output_dir / "dry_run" / "rollout_report.json", "rollout_report.json"),
        (output_dir / "eval" / "eval_report.json", "eval_report.json"),
        (output_dir / "compare" / "compare_report.json", "compare_report.json"),
    ]:
        if src.exists():
            _shutil.copy2(src, reports_dir / dst_name)

    # Copy traces.
    traces_dir = publish_dir / "traces"
    traces_dir.mkdir(exist_ok=True)
    for src, dst_name in [
        (output_dir / "export_trace.jsonl", "export_trace.jsonl"),
        (output_dir / "eval" / "eval_trace.jsonl", "eval_trace.jsonl"),
        (output_dir / "compare" / "compare_trace.jsonl", "compare_trace.jsonl"),
    ]:
        if src.exists():
            _shutil.copy2(src, traces_dir / dst_name)

    # Write README.
    model_id = report.get("artifact", {}).get("model_id", "unknown")
    source = report.get("source", {})
    readme = f"""# {model_id}
CoreAI-backed LeRobot policy artifact.

Source policy: {source.get("torch_policy_path", "unknown")}
Policy type: {source.get("policy_type", "unknown")}
Robot type: {source.get("robot_type", "unknown")}
Runtime: CoreAI
Default mode: dry_run

## Verification
- Manifest valid: {report.get("verification", {}).get("manifest_valid", False)}
- Dry-run: {report.get("verification", {}).get("dry_run", {}).get("ran", False)}
- Dataset eval: {report.get("verification", {}).get("eval", {}).get("ran", False)}
- Action parity: {report.get("claims", {}).get("proves_numeric_action_fidelity", False)}

## Claims
- Numeric fidelity: {report.get("claims", {}).get("proves_numeric_action_fidelity", False)}
- Task success: false
- Physical robot safety: false

No robot commands were sent during verification.
"""
    (publish_dir / "README.md").write_text(readme)
