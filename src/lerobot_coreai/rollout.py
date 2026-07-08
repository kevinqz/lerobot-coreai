# rollout.py — fixture-based dry-run rollout (v0.3).
#
# dry_run: loads observation from fixture, calls runner, generates action, writes report.
# No robot connection. No motor commands. No hardware.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import CoreAIPolicyError
from .fixtures import load_observation_fixture
from .policy import CoreAIPolicy
from .reports import build_failure_report, build_success_report, save_json
from .safety import ensure_mode_supported_for_v03
from .trace import TraceWriter
from .validation import validate_robot_type


@dataclass
class DryRunRolloutConfig:
    policy_path: str
    robot_type: str | None = None
    fixture_path: Path = Path("observation.json")
    runner_url: str = "unix:///tmp/coreai-runner.sock"
    output_dir: Path = Path("runs/dry-run")
    strict_observation_keys: bool = False
    keep_temp_files: bool = False
    confirm_real_robot_actuation: bool = False
    overwrite: bool = False


@dataclass
class DryRunRolloutResult:
    ok: bool
    output_dir: Path
    action_path: Path
    observation_path: Path
    report_path: Path
    trace_path: Path
    report: dict[str, Any] = field(default_factory=dict)


def run_dry_run_rollout(config: DryRunRolloutConfig) -> DryRunRolloutResult:
    """Execute a fixture-based dry-run rollout.

    Flow:
    1. Check mode is dry_run (block shadow/sim/real)
    2. Prepare output_dir
    3. Load policy (with runner validation)
    4. Validate robot type
    5. Load fixture → save observation.json
    6. Call predict_action → save action.json
    7. Build + save rollout_report.json
    8. Write trace.jsonl

    Never sends robot commands.
    """
    output_dir = Path(config.output_dir)
    action_path = output_dir / "action.json"
    observation_path = output_dir / "observation.json"
    report_path = output_dir / "rollout_report.json"
    trace_path = output_dir / "trace.jsonl"

    # Check mode.
    ensure_mode_supported_for_v03("dry_run", confirm_real_robot_actuation=config.confirm_real_robot_actuation)

    # Prepare output dir.
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Output directory not empty: {output_dir}. Use --overwrite to replace."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    trace = TraceWriter(trace_path)
    trace.write("rollout.started", {"mode": "dry_run", "policy": config.policy_path})

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
        trace.write("policy.loaded", {
            "policy_type": policy.policy_type,
            "robot_type": policy.robot_type,
            "model_id": policy.manifest.model_id,
        })
        trace.write("runner.checked")

        # Validate robot type.
        stage = "robot_type.validation"
        validate_robot_type(config.robot_type, policy.manifest)
        trace.write("robot_type.validated", {"requested": config.robot_type})

        # Load fixture.
        stage = "fixture.load"
        observation = load_observation_fixture(config.fixture_path)
        trace.write("observation.loaded", {"source": str(config.fixture_path), "keys": list(observation.keys())})

        # Save observation.
        stage = "observation.save"
        save_json(observation_path, observation)

        # Predict action.
        stage = "runner.predict"
        result = policy.predict_action(observation, return_metadata=True)
        action = result["action"]
        metadata = result.get("metadata", {})
        trace.write("action.generated", {"shape": _safe_shape(action)})

        # Save action.
        stage = "action.save"
        save_json(action_path, result)

        # Build success report.
        files = {
            "observation": "observation.json",
            "action": "action.json",
            "trace": "trace.jsonl",
            "report": "rollout_report.json",
        }
        report = build_success_report(
            policy_path=config.policy_path,
            source_repo_id=policy.manifest.policy_source_repo_id,
            policy_type=policy.policy_type,
            model_id=policy.manifest.model_id,
            robot_type=config.robot_type or policy.robot_type,
            runner_url=config.runner_url,
            runner_timing=metadata.get("timing"),
            parity_passed=policy.parity_passed,
            fixture_source=str(config.fixture_path),
            observation_keys=list(observation.keys()),
            action=action,
            files=files,
        )
        save_json(report_path, report)
        stage = "report.write"
        trace.write("files.written", {"files": list(files.keys())})
        trace.write("rollout.completed", {"ok": True})

        return DryRunRolloutResult(
            ok=True,
            output_dir=output_dir,
            action_path=action_path,
            observation_path=observation_path,
            report_path=report_path,
            trace_path=trace_path,
            report=report,
        )

    except Exception as e:
        # Build failure report.
        error_type = type(e).__name__
        error_msg = str(e)

        # Try to save failure report.
        try:
            fail_report = build_failure_report(
                policy_path=config.policy_path,
                robot_type=config.robot_type,
                mode="dry_run",
                error_type=error_type,
                error_message=error_msg,
                stage=stage,
                runner_url=config.runner_url,
            )
            save_json(report_path, fail_report)
        except Exception:
            pass  # best-effort

        trace.write("rollout.failed", {"error": error_type, "message": error_msg})
        raise


def _safe_shape(value: Any) -> list[int] | None:
    if isinstance(value, (list, tuple)) and len(value) > 0:
        inner = _safe_shape(value[0])
        return [len(value)] + (inner or [])
    return None
