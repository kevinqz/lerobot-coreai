# cli.py — command-line interface for lerobot-coreai (spec §12).
#
# The CLI is shaped like LeRobot workflows. The only new word is 'coreai'.
#
# MVP v0.1 implements: inspect, doctor.
# Future versions add: export, eval, rollout, serve, compare.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .catalog import list_lerobot_policies
from .compatibility.versions import check_lerobot_compatibility, get_installed_lerobot_version
from .errors import CoreAIPolicyError, DownloadError, ManifestError
from .manifest import load_manifest
from .rollout import DryRunRolloutConfig, run_dry_run_rollout
from .eval import EvalConfig, run_lerobot_dataset_eval
from .compare import CompareConfig, run_lerobot_policy_compare
from .export import ExportConfig, run_coreai_export_pipeline
from .shadow import ShadowConfig, run_shadow_mode
from .sim import SimConfig, run_sim_mode


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except ManifestError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 3
    except CoreAIPolicyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lerobot-coreai",
        description="Apple CoreAI runtime backend for LeRobot policies.",
    )
    parser.add_argument("--version", action="version", version=f"lerobot-coreai {__version__}")

    sub = parser.add_subparsers(dest="command")

    # --- inspect (spec §12.2) ---
    p_inspect = sub.add_parser("inspect", help="Inspect a CoreAI-backed LeRobot policy")
    p_inspect.add_argument("--policy.path", dest="policy_path", required=True,
                           help="HF repo id of the CoreAI artifact")
    p_inspect.add_argument("--json", action="store_true", help="Output as JSON")
    p_inspect.set_defaults(func=cmd_inspect)

    # --- doctor (spec §12.6) ---
    p_doctor = sub.add_parser("doctor", help="Diagnose policy/robot/runtime compatibility")
    p_doctor.add_argument("--policy.path", dest="policy_path",
                          help="HF repo id of the CoreAI artifact")
    p_doctor.add_argument("--robot.type", dest="robot_type",
                          help="Robot type to check against policy metadata")
    p_doctor.add_argument("--runner.url", dest="runner_url",
                          help="coreai-runner URL to check (e.g. http://127.0.0.1:8710)")
    p_doctor.add_argument("--require-runner", dest="require_runner", action="store_true",
                          help="Exit non-zero if runner is not reachable")
    p_doctor.set_defaults(func=cmd_doctor)

    # --- list (spec §P3) — query the catalog for LeRobot policies ---
    p_list = sub.add_parser("list", help="List CoreAI-backed LeRobot policies from the catalog")
    p_list.add_argument("--robot.type", dest="robot_type",
                        help="Filter by robot type (e.g. so100, so101, aloha)")
    p_list.add_argument("--policy.type", dest="policy_type",
                        help="Filter by policy type (e.g. act, pi0, diffusion, evo1)")
    p_list.add_argument("--status", dest="status",
                        help="Filter by parity status (e.g. action_parity_passed)")
    p_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_list.set_defaults(func=cmd_list)

    # --- predict (v0.2) — one observation in, one action out ---
    p_predict = sub.add_parser("predict", help="Predict an action from a single observation (v0.2)")
    p_predict.add_argument("--policy.path", dest="policy_path", required=True,
                           help="HF repo id of the CoreAI artifact")
    p_predict.add_argument("--observation", dest="observation", required=True,
                           help="Path to observation fixture JSON")
    p_predict.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock",
                           help="coreai-runner URL (default: unix socket)")
    p_predict.add_argument("--output", dest="output",
                           help="Write action JSON to file (default: stdout)")
    p_predict.add_argument("--json", action="store_true", help="Output as JSON")
    p_predict.add_argument("--metadata", action="store_true", help="Include metadata in output")
    p_predict.set_defaults(func=cmd_predict)

    # --- export (spec §12.3) — v0.4 ---
    # --- export (spec §12.3) — v0.6 ---
    p_export = sub.add_parser("export", help="Export/verify/package a LeRobot policy as CoreAI artifact (v0.6)")
    p_export.add_argument("--torch.policy.path", dest="torch_policy_path", required=True)
    p_export.add_argument("--policy.type", dest="policy_type")
    p_export.add_argument("--robot.type", dest="robot_type")
    p_export.add_argument("--dataset.repo_id", dest="dataset_repo_id")
    p_export.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock")
    p_export.add_argument("--output-dir", dest="output_dir", required=True)
    p_export.add_argument("--model-id", dest="model_id")
    p_export.add_argument("--output.repo_id", dest="output_repo_id")
    p_export.add_argument("--artifact-name", dest="artifact_name")
    p_export.add_argument("--fabric.config", dest="fabric_config")
    p_export.add_argument("--fabric.profile", dest="fabric_profile")
    p_export.add_argument("--fabric.target", dest="fabric_target", default="coreai")
    p_export.add_argument("--skip-fabric", action="store_true")
    p_export.add_argument("--existing-artifact", dest="existing_artifact")
    p_export.add_argument("--verify-runner", action="store_true")
    p_export.add_argument("--dry-run-fixture", dest="dry_run_fixture")
    p_export.add_argument("--eval-max-frames", dest="eval_max_frames", type=int, default=0)
    p_export.add_argument("--compare-max-frames", dest="compare_max_frames", type=int, default=0)
    p_export.add_argument("--compare-tolerance.cosine", dest="compare_tolerance_cosine", type=float, default=0.999)
    p_export.add_argument("--compare-tolerance.max-mae", dest="compare_tolerance_max_mae", type=float, default=1e-4)
    p_export.add_argument("--compare-tolerance.mean-mae", dest="compare_tolerance_mean_mae", type=float, default=1e-5)
    p_export.add_argument("--publish-ready", action="store_true")
    p_export.add_argument("--overwrite", action="store_true")
    p_export.add_argument("--fail-fast", action="store_true")
    p_export.add_argument("--json", action="store_true")
    p_export.set_defaults(func=cmd_export)

    # --- eval (spec §12.4) — v0.4 ---
    # --- eval (spec §12.4) — v0.4 ---
    p_eval = sub.add_parser("eval", help="Evaluate a CoreAI policy on a LeRobotDataset (v0.4)")
    p_eval.add_argument("--policy.path", dest="policy_path", required=True)
    p_eval.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    p_eval.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock")
    p_eval.add_argument("--robot.type", dest="robot_type")
    p_eval.add_argument("--max-frames", dest="max_frames", type=int, default=32)
    p_eval.add_argument("--start-index", dest="start_index", type=int, default=0)
    p_eval.add_argument("--stride", dest="stride", type=int, default=1)
    p_eval.add_argument("--episodes", dest="episodes",
                        help="Comma-separated episode indices (e.g. 0,1,2)")
    p_eval.add_argument("--dataset.root", dest="dataset_root")
    p_eval.add_argument("--dataset.revision", dest="dataset_revision")
    p_eval.add_argument("--no-download-videos", dest="download_videos", action="store_false", default=True)
    p_eval.add_argument("--video-backend", dest="video_backend")
    p_eval.add_argument("--output-dir", dest="output_dir")
    p_eval.add_argument("--overwrite", action="store_true")
    p_eval.add_argument("--strict", action="store_true")
    p_eval.add_argument("--fail-fast", action="store_true")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    # --- rollout (spec §12.5) — v0.2+ ---
    p_rollout = sub.add_parser("rollout", help="Run fixture-based dry-run rollout (v0.3)")
    p_rollout.add_argument("--policy.path", dest="policy_path", required=True)
    p_rollout.add_argument("--robot.type", dest="robot_type")
    p_rollout.add_argument("--mode", choices=["dry_run", "shadow", "sim", "real"], default="dry_run")
    p_rollout.add_argument("--fixture", dest="fixture",
                           help="Path to observation fixture JSON (required for dry_run)")
    p_rollout.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock")
    p_rollout.add_argument("--output-dir", dest="output_dir", default=None,
                           help="Output directory (default: runs/<policy>-dry-run)")
    p_rollout.add_argument("--strict", dest="strict", action="store_true")
    p_rollout.add_argument("--keep-temp-files", dest="keep_temp_files", action="store_true")
    p_rollout.add_argument("--overwrite", dest="overwrite", action="store_true")
    p_rollout.add_argument("--confirm-real-robot-actuation", dest="confirm_real", action="store_true")
    p_rollout.add_argument("--json", action="store_true")
    p_rollout.set_defaults(func=cmd_rollout)

    # --- shadow (v0.7) — motor-blocked observation loop ---
    p_shadow = sub.add_parser("shadow", help="Run motor-blocked shadow mode (v0.7)")
    p_shadow.add_argument("--policy.path", dest="policy_path", required=True,
                          help="HF repo id of the CoreAI artifact")
    p_shadow.add_argument("--observation-source", dest="observation_source", required=True,
                          choices=["fixture", "fixtures", "folder", "camera"],
                          help="Where to read observations from")
    p_shadow.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock",
                          help="coreai-runner URL (default: unix socket)")
    p_shadow.add_argument("--output-dir", dest="output_dir", required=True,
                          help="Output directory for reports/logs")
    p_shadow.add_argument("--robot.type", dest="robot_type")
    # Observation source args.
    p_shadow.add_argument("--fixture", dest="fixture",
                          help="Single observation fixture JSON (for --observation-source fixture)")
    p_shadow.add_argument("--fixtures-dir", dest="fixtures_dir",
                          help="Directory of ordered fixture JSONs (for --observation-source fixtures)")
    p_shadow.add_argument("--frames-dir", dest="frames_dir",
                          help="Directory of image frames (for --observation-source folder)")
    p_shadow.add_argument("--image-key", dest="image_key", default="observation.images.wrist",
                          help="Observation key for image frames (default: observation.images.wrist)")
    p_shadow.add_argument("--state-json", dest="state_json",
                          help="JSON array file for observation.state")
    p_shadow.add_argument("--state-vector", dest="state_vector",
                          help="Comma-separated floats for observation.state")
    p_shadow.add_argument("--task", dest="task", help="Task text to include in each observation")
    # Camera source args (v0.7.1).
    p_shadow.add_argument("--camera.index", dest="camera_index", type=int, default=0,
                          help="Camera device index (default: 0)")
    p_shadow.add_argument("--camera.width", dest="camera_width", type=int,
                          help="Requested camera frame width")
    p_shadow.add_argument("--camera.height", dest="camera_height", type=int,
                          help="Requested camera frame height")
    p_shadow.add_argument("--camera.fps", dest="camera_fps", type=float,
                          help="Requested camera FPS")
    # Loop args.
    p_shadow.add_argument("--max-steps", dest="max_steps", type=int, default=32)
    p_shadow.add_argument("--duration-seconds", dest="duration_seconds", type=float)
    p_shadow.add_argument("--fps", dest="fps", type=float, default=10.0)
    p_shadow.add_argument("--warmup-steps", dest="warmup_steps", type=int, default=0)
    p_shadow.add_argument("--strict", dest="strict", action="store_true")
    p_shadow.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    p_shadow.add_argument("--overwrite", dest="overwrite", action="store_true")
    p_shadow.add_argument("--metadata", dest="metadata", action="store_true")
    # Adapter args (v0.7.2).
    p_shadow.add_argument("--adapter.image-key", dest="adapter_image_key",
                          help="Override image observation key for adapter")
    p_shadow.add_argument("--adapter.image-map", dest="adapter_image_map",
                          help="Image alias mapping: alias1=key1,alias2=key2")
    p_shadow.add_argument("--adapter.require-task", dest="adapter_require_task",
                          action="store_true", help="Require task key in observations")
    p_shadow.add_argument("--adapter.require-state", dest="adapter_require_state",
                          action="store_true", help="Require observation.state in observations")
    p_shadow.add_argument("--adapter.required-keys", dest="adapter_required_keys",
                          help="Comma-separated required observation keys")
    p_shadow.add_argument("--adapter.drop-unknown-keys", dest="adapter_drop_unknown_keys",
                          action="store_true", help="Drop keys not in manifest")
    # Live metrics / quality args (v0.7.2).
    p_shadow.add_argument("--live", dest="live", action="store_true",
                          help="Print live metrics per step")
    p_shadow.add_argument("--live-every", dest="live_every", type=int, default=1,
                          help="Print live metrics every N steps (default: 1)")
    p_shadow.add_argument("--quality.max-runner-p95-ms", dest="quality_max_runner_p95_ms", type=float)
    p_shadow.add_argument("--quality.max-loop-p95-ms", dest="quality_max_loop_p95_ms", type=float)
    p_shadow.add_argument("--quality.max-error-rate", dest="quality_max_error_rate", type=float, default=0.0)
    p_shadow.add_argument("--quality.min-effective-fps", dest="quality_min_effective_fps", type=float)
    p_shadow.add_argument("--quality.fail-on-quality", dest="quality_fail_on_quality", action="store_true",
                          help="Fail the run if quality gates don't pass")
    p_shadow.add_argument("--json", action="store_true")
    p_shadow.set_defaults(func=cmd_shadow)

    # --- sim (v0.8) — simulator-only action egress ---
    p_sim = sub.add_parser("sim", help="Run simulator-only sim mode (v0.8)")
    p_sim.add_argument("--policy.path", dest="policy_path", required=True,
                       help="HF repo id of the CoreAI artifact")
    p_sim.add_argument("--env.type", dest="env_type", required=True,
                       help="Simulator environment type (v0.8.0: fake, replay; gym/lerobot/pusht reserved for v0.8.1)")
    p_sim.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock",
                       help="coreai-runner URL (default: unix socket)")
    p_sim.add_argument("--output-dir", dest="output_dir", required=True,
                       help="Output directory for reports/logs")
    p_sim.add_argument("--robot.type", dest="robot_type")
    # Environment args.
    p_sim.add_argument("--env.config", dest="env_config",
                       help="Environment config JSON (for --env.type replay)")
    p_sim.add_argument("--env.render", dest="env_render", action="store_true")
    p_sim.add_argument("--env.record-video", dest="env_record_video", action="store_true")
    p_sim.add_argument("--env.video-dir", dest="env_video_dir")
    # Observation args.
    p_sim.add_argument("--task", dest="task", help="Task text to include in each observation")
    p_sim.add_argument("--state-vector", dest="state_vector",
                       help="Comma-separated floats for observation.state")
    p_sim.add_argument("--image-key", dest="image_key", default="observation.images.wrist",
                       help="Observation key for image frames (default: observation.images.wrist)")
    # Loop args.
    p_sim.add_argument("--episodes", dest="episodes", type=int, default=1)
    p_sim.add_argument("--max-steps-per-episode", dest="max_steps_per_episode", type=int, default=300)
    p_sim.add_argument("--seed", dest="seed", type=int)
    p_sim.add_argument("--fps", dest="fps", type=float, default=0.0)
    p_sim.add_argument("--strict", dest="strict", action="store_true")
    p_sim.add_argument("--fail-fast", dest="fail_fast", action="store_true")
    p_sim.add_argument("--overwrite", dest="overwrite", action="store_true")
    p_sim.add_argument("--live", dest="live", action="store_true",
                       help="Print live metrics per step")
    p_sim.add_argument("--live-every", dest="live_every", type=int, default=1,
                       help="Print live metrics every N steps (default: 1)")
    p_sim.add_argument("--confirm-sim-egress", dest="confirm_sim_egress", action="store_true",
                       help="Confirm that actions may be sent to the simulator (required)")
    p_sim.add_argument("--json", action="store_true")
    p_sim.set_defaults(func=cmd_sim)

    # --- compare (spec §12.7) — v0.3 ---
    p_compare = sub.add_parser("compare", help="Compare PyTorch vs CoreAI action parity on LeRobotDataset (v0.5)")
    p_compare.add_argument("--torch.policy.path", dest="torch_policy_path", required=True)
    p_compare.add_argument("--torch.policy.type", dest="torch_policy_type")
    p_compare.add_argument("--coreai.policy.path", dest="coreai_policy_path", required=True)
    p_compare.add_argument("--dataset.repo_id", dest="dataset_repo_id", required=True)
    p_compare.add_argument("--runner.url", dest="runner_url", default="unix:///tmp/coreai-runner.sock")
    p_compare.add_argument("--robot.type", dest="robot_type")
    p_compare.add_argument("--max-frames", dest="max_frames", type=int, default=32)
    p_compare.add_argument("--start-index", dest="start_index", type=int, default=0)
    p_compare.add_argument("--stride", dest="stride", type=int, default=1)
    p_compare.add_argument("--episodes", dest="episodes")
    p_compare.add_argument("--dataset.root", dest="dataset_root")
    p_compare.add_argument("--dataset.revision", dest="dataset_revision")
    p_compare.add_argument("--no-download-videos", dest="download_videos", action="store_false", default=True)
    p_compare.add_argument("--video-backend", dest="video_backend")
    p_compare.add_argument("--output-dir", dest="output_dir")
    p_compare.add_argument("--overwrite", action="store_true")
    p_compare.add_argument("--strict", action="store_true")
    p_compare.add_argument("--fail-fast", action="store_true")
    p_compare.add_argument("--tolerance.cosine", dest="tolerance_cosine", type=float, default=0.999)
    p_compare.add_argument("--tolerance.max-mae", dest="tolerance_max_mae", type=float, default=1e-4)
    p_compare.add_argument("--tolerance.mean-mae", dest="tolerance_mean_mae", type=float, default=1e-5)
    p_compare.add_argument("--save-actions", action="store_true")
    p_compare.add_argument("--reset-each-frame", action="store_true")
    p_compare.add_argument("--json", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    # --- serve (spec §12, serve) — v0.2 ---
    p_serve = sub.add_parser("serve", help="Start or connect to coreai-runner (future)")
    p_serve.set_defaults(func=cmd_not_implemented)

    return parser


def cmd_not_implemented(args: argparse.Namespace) -> int:
    print(
        f"'{args.command}' is not implemented in v0.8. "
        f"Available commands: inspect, doctor, list, predict, rollout --mode dry_run, shadow, sim, eval, compare, export.",
        file=sys.stderr,
    )
    return 1


# MARK: - inspect (spec §12.2)

def cmd_inspect(args: argparse.Namespace) -> int:
    """Inspect a CoreAI-backed LeRobot policy."""
    manifest = load_manifest(args.policy_path)

    if args.json:
        import json
        print(json.dumps(manifest.raw, indent=2))
        return 0

    # Pretty-print (spec §12.2 output format).
    print(f"Policy: {manifest.policy_type.upper()}")
    print(f"Runtime: CoreAI")
    print(f"Artifact: {manifest.policy_repo_id}")
    print(f"Source: {manifest.policy_source_repo_id}")
    print(f"Robot type: {manifest.robot_type}")
    if manifest.robot_fps:
        print(f"Control rate: {manifest.robot_fps} fps")
    print(f"LeRobot version: {manifest.framework_version}")
    print()
    print("Observation features:")
    for name, feat in manifest.observation_features.items():
        shape_str = f"[{', '.join(str(d) for d in feat.shape)}]" if feat.shape else "?"
        print(f"  - {name}: {shape_str}")
    print("Action features:")
    for name, feat in manifest.action_features.items():
        shape_str = f"[{', '.join(str(d) for d in feat.shape)}]" if feat.shape else "?"
        print(f"  - {name}: {shape_str}")
    print()
    print(f"CoreAI parity: {manifest.evaluation_status}")
    if manifest.evaluation_min_chunk_cosine is not None:
        print(f"  min chunk cosine: {manifest.evaluation_min_chunk_cosine}")
    if manifest.evaluation_max_action_mae is not None:
        print(f"  max action MAE: {manifest.evaluation_max_action_mae}")
    if manifest.graphs:
        print(f"Graphs: {', '.join(g.name for g in manifest.graphs)}")
    if manifest.host_loop_required:
        print(f"Host loop: {manifest.host_loop_type} ({manifest.host_loop_solver}, {manifest.host_loop_num_steps} steps)")
    print()
    print(f"Default mode: {manifest.default_mode}")
    print(f"Recommended next step: rollout --mode dry_run")

    return 0


# MARK: - doctor (spec §12.6)

def cmd_doctor(args: argparse.Namespace) -> int:
    """Diagnose policy/robot/runtime compatibility."""
    checks: list[tuple[bool, str]] = []

    # Check 1: lerobot-coreai version
    checks.append((True, f"lerobot-coreai {__version__} installed"))

    # Check 2: LeRobot version
    lerobot_ver = get_installed_lerobot_version()
    if lerobot_ver:
        status, msg = check_lerobot_compatibility(allow_unsupported=True)
        checks.append((status == "supported", f"LeRobot {lerobot_ver} — {msg}"))
    else:
        checks.append((True, "LeRobot not installed (metadata-only mode)"))

    # Check 3-8: manifest checks (only if policy.path given)
    manifest = None
    if args.policy_path:
        try:
            manifest = load_manifest(args.policy_path)
            checks.append((True, f"CoreAI artifact found: {args.policy_path}"))
            checks.append((True, f"lerobot-coreai.json found and valid"))
        except ManifestError as e:
            checks.append((False, f"Manifest error: {e}"))
        except DownloadError as e:
            checks.append((False, f"Download error: {e}"))

    if manifest:
        # Robot type match
        if args.robot_type:
            if args.robot_type == manifest.robot_type:
                checks.append((True, f"Robot type matches: {args.robot_type}"))
            else:
                checks.append((False,
                    f"Robot type mismatch: policy expects {manifest.robot_type}, got {args.robot_type}"))

        # Parity
        if manifest.parity_passed:
            checks.append((True, f"Action parity passed"))
        else:
            checks.append((False, f"Action parity: {manifest.evaluation_status}"))

        # Default mode
        checks.append((True, f"Default mode: {manifest.default_mode}"))

    # Runner check (v0.2): if --runner.url provided, check health/capabilities.
    runner_url = getattr(args, "runner_url", None)
    require_runner = getattr(args, "require_runner", False)

    if runner_url:
        from .runner import RunnerClient
        from .errors import RunnerError
        try:
            rc = RunnerClient(runner_url)
            health = rc.health()
            checks.append((True, f"coreai-runner reachable at {runner_url} (status: {health.status})"))
            caps = rc.capabilities()
            if caps.supports_action:
                checks.append((True, "coreai-runner supports runtime_kind=action"))
            else:
                checks.append((False, "coreai-runner does NOT support runtime_kind=action"))
            if caps.supports_host_loop:
                checks.append((True, "coreai-runner supports host_loop"))
            rc.close()
        except RunnerError as e:
            checks.append((False, f"coreai-runner check failed: {e}"))
    elif require_runner:
        # --require-runner without --runner.url: check the default socket.
        from .runner import RunnerClient
        from .errors import RunnerError
        default_url = "unix:///tmp/coreai-runner.sock"
        try:
            rc = RunnerClient(default_url)
            health = rc.health()
            checks.append((True, f"coreai-runner reachable at {default_url} (default, status: {health.status})"))
            caps = rc.capabilities()
            if caps.supports_action:
                checks.append((True, "coreai-runner supports runtime_kind=action"))
            else:
                checks.append((False, "coreai-runner does NOT support runtime_kind=action"))
            rc.close()
        except RunnerError as e:
            checks.append((False, f"coreai-runner not reachable at {default_url}: {e}"))
    else:
        checks.append((True, "coreai-runner check: skipped (pass --runner.url to check)"))

    # Print results
    print("lerobot-coreai doctor")
    print("=" * 50)
    for ok, msg in checks:
        symbol = "✓" if ok else "✗"
        print(f"{symbol} {msg}")
    print("=" * 50)

    all_ok = all(ok for ok, _ in checks)
    if all_ok:
        print("All checks passed.")
    else:
        print("Some checks failed.")

    return 0 if all_ok else 1


# MARK: - list (query catalog for LeRobot policies)

def cmd_list(args: argparse.Namespace) -> int:
    """List CoreAI-backed LeRobot policies from the catalog."""
    policies = list_lerobot_policies()

    # Apply filters
    if args.robot_type:
        policies = [p for p in policies if p.get("robot_type") == args.robot_type]
    if args.policy_type:
        policies = [p for p in policies if p.get("policy_type") == args.policy_type]
    if args.status:
        policies = [p for p in policies if args.status in (p.get("status") or "")]

    if args.json:
        import json
        print(json.dumps({"policies": policies}, indent=2))
        return 0

    if not policies:
        print("No LeRobot CoreAI policies found matching the criteria.")
        return 0

    print(f"LeRobot CoreAI policies ({len(policies)}):")
    print()
    print(f"{'REPO_ID':<45s} {'TYPE':<12s} {'ROBOT':<10s} {'STATUS'}")
    print("-" * 85)
    for p in policies:
        repo = p.get("repo_id") or p.get("catalog_model_id") or "?"
        ptype = p.get("policy_type", "?")
        robot = p.get("robot_type", "?")
        status = p.get("status", "?")
        print(f"{repo:<45s} {ptype:<12s} {robot:<10s} {status}")

    return 0


# MARK: - predict (v0.2 — one observation in, one action out)

def cmd_predict(args: argparse.Namespace) -> int:
    """Predict an action from a single observation fixture.

    Loads the observation JSON, calls select_action via the runner, and prints the action.
    Never connects to a physical robot.
    """
    import json
    from pathlib import Path
    from .policy import CoreAIPolicy
    from .errors import CoreAIPolicyError

    # Load observation fixture (supports flat + typed formats with path resolution).
    from .fixtures import load_observation_fixture

    # Load policy with runner.
    try:
        observation = load_observation_fixture(args.observation)
        policy = CoreAIPolicy.from_pretrained(
            args.policy_path,
            runner_url=args.runner_url,
        )
        result = policy.predict_action(observation, return_metadata=args.metadata)
    except CoreAIPolicyError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    # Output action.
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(f"Action written to {args.output}")
    elif args.json:
        print(json.dumps(result, indent=2))
    else:
        action = result["action"]
        print(f"Action shape: [{len(action)}{''.join(f', {len(action[0])}' if action and isinstance(action[0], list) else '')}]")
        print(f"Action: {json.dumps(action)}")

    return 0


# MARK: - rollout (v0.3 — fixture-based dry_run)

def cmd_rollout(args: argparse.Namespace) -> int:
    """Run a fixture-based dry-run rollout."""
    from .safety import ensure_mode_supported_for_v03
    from .errors import CoreAIPolicyError, SafetyError

    mode = args.mode

    # Block non-dry_run modes.
    try:
        ensure_mode_supported_for_v03(mode, confirm_real_robot_actuation=args.confirm_real)
    except SafetyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if mode == "dry_run" and not args.fixture:
        print("Error: --fixture is required for dry_run mode.", file=sys.stderr)
        return 1

    output_dir = args.output_dir or f"runs/{args.policy_path.split('/')[-1]}-dry-run"

    config = DryRunRolloutConfig(
        policy_path=args.policy_path,
        robot_type=args.robot_type,
        fixture_path=Path(args.fixture),
        runner_url=args.runner_url,
        output_dir=Path(output_dir),
        strict_observation_keys=args.strict,
        keep_temp_files=args.keep_temp_files,
        overwrite=args.overwrite,
        confirm_real_robot_actuation=args.confirm_real,
    )

    print(f"lerobot-coreai rollout")
    print("=" * 50)
    print(f"Policy: {args.policy_path}")
    print(f"Mode: {mode}")
    print(f"Robot type: {args.robot_type or '(from manifest)'}")
    print(f"Runner: {args.runner_url}")

    try:
        result = run_dry_run_rollout(config)
    except CoreAIPolicyError as e:
        print(f"\n✗ Rollout failed: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        report_path = Path(output_dir) / "rollout_report.json"
        if report_path.exists():
            print(f"Failure report: {report_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(result.report, indent=2))
        return 0

    print()
    print(f"✓ Manifest valid")
    print(f"✓ Runner reachable")
    print(f"✓ Runner supports runtime_kind=action")
    print(f"✓ Observation fixture loaded")
    print(f"✓ Action generated")
    print(f"✓ No robot commands sent")
    print()
    print("Files:")
    print(f"  action:      {result.action_path}")
    print(f"  observation: {result.observation_path}")
    print(f"  trace:       {result.trace_path}")
    print(f"  report:      {result.report_path}")
    print("=" * 50)
    print("Dry-run completed successfully.")

    return 0


# MARK: - eval (v0.4 — LeRobotDataset replay/eval)

def cmd_eval(args: argparse.Namespace) -> int:
    """Evaluate a CoreAI policy on a LeRobotDataset."""
    from .errors import CoreAIPolicyError

    # Parse episodes.
    episodes = None
    if args.episodes:
        episodes = [int(e.strip()) for e in args.episodes.split(",")]

    output_dir = args.output_dir or f"runs/{args.policy_path.split('/')[-1]}-eval"

    config = EvalConfig(
        policy_path=args.policy_path,
        dataset_repo_id=args.dataset_repo_id,
        runner_url=args.runner_url,
        output_dir=Path(output_dir),
        robot_type=args.robot_type,
        max_frames=args.max_frames,
        start_index=args.start_index,
        stride=args.stride,
        episodes=episodes,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        dataset_revision=args.dataset_revision,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
        strict_observation_keys=args.strict,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
    )

    if not args.json:
        print(f"lerobot-coreai eval")
        print("=" * 50)
        print(f"Policy: {args.policy_path}")
        print(f"Dataset: {args.dataset_repo_id}")
        print(f"Runner: {args.runner_url}")
        print(f"Max frames: {args.max_frames}")

    try:
        result = run_lerobot_dataset_eval(config)
    except CoreAIPolicyError as e:
        print(f"\n✗ Eval failed: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(result.report, indent=2))
        return 0

    m = result.report.get("metrics", {})
    print()
    print(f"✓ Eval completed")
    print(f"  frames processed: {m.get('frames_processed', 0)}")
    print(f"  actions generated: {m.get('actions_generated', 0)}")
    print(f"  actions failed: {m.get('actions_failed', 0)}")
    if m.get("mean_total_ms"):
        print(f"  mean inference: {m['mean_total_ms']:.1f}ms")
    print(f"  No robot commands sent")
    print()
    print("Files:")
    print(f"  actions: {result.actions_path}")
    print(f"  trace:   {result.trace_path}")
    print(f"  report:  {result.report_path}")
    print("=" * 50)

    return 0 if result.ok else 1


# MARK: - shadow (v0.7 — motor-blocked observation loop)

def cmd_shadow(args: argparse.Namespace) -> int:
    """Run motor-blocked shadow mode: observe, generate actions, block all egress."""
    from .errors import CoreAIPolicyError
    from .shadow_quality import ShadowQualityConfig

    # Parse state-vector (comma-separated floats) if provided.
    state_vector = None
    if args.state_vector:
        state_vector = [float(v.strip()) for v in args.state_vector.split(",")]

    # Parse adapter image map (alias1=key1,alias2=key2).
    adapter_image_keys = None
    if getattr(args, "adapter_image_map", None):
        adapter_image_keys = {}
        for pair in args.adapter_image_map.split(","):
            if "=" in pair:
                alias, key = pair.split("=", 1)
                adapter_image_keys[alias.strip()] = key.strip()

    # Parse adapter required keys.
    required_keys = None
    if getattr(args, "adapter_required_keys", None):
        required_keys = [k.strip() for k in args.adapter_required_keys.split(",")]

    # Build quality config if any quality args provided.
    quality_config = None
    if any(getattr(args, attr) is not None for attr in [
        "quality_max_runner_p95_ms", "quality_max_loop_p95_ms", "quality_min_effective_fps"
    ]) or getattr(args, "quality_max_error_rate", 0.0) != 0.0:
        quality_config = ShadowQualityConfig(
            max_runner_p95_ms=args.quality_max_runner_p95_ms,
            max_loop_p95_ms=args.quality_max_loop_p95_ms,
            max_error_rate=args.quality_max_error_rate,
            min_effective_fps=args.quality_min_effective_fps,
        )

    config = ShadowConfig(
        policy_path=args.policy_path,
        runner_url=args.runner_url,
        output_dir=Path(args.output_dir),
        observation_source=args.observation_source,
        robot_type=args.robot_type,
        fixture=Path(args.fixture) if args.fixture else None,
        fixtures_dir=Path(args.fixtures_dir) if args.fixtures_dir else None,
        frames_dir=Path(args.frames_dir) if args.frames_dir else None,
        image_key=args.image_key,
        state_json=Path(args.state_json) if args.state_json else None,
        state_vector=state_vector,
        task=args.task,
        camera_index=args.camera_index,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        max_steps=args.max_steps,
        duration_seconds=args.duration_seconds,
        fps=args.fps,
        warmup_steps=args.warmup_steps,
        strict_observation_keys=args.strict,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
        adapter_image_keys=adapter_image_keys,
        require_task=getattr(args, "adapter_require_task", False),
        require_state=getattr(args, "adapter_require_state", False),
        required_keys=required_keys,
        drop_unknown_keys=getattr(args, "adapter_drop_unknown_keys", False),
        live=getattr(args, "live", False),
        live_every=getattr(args, "live_every", 1),
        quality_config=quality_config,
        fail_on_quality=getattr(args, "quality_fail_on_quality", False),
    )

    if not args.json:
        print(f"lerobot-coreai shadow")
        print("=" * 50)
        print(f"Policy: {args.policy_path}")
        print(f"Mode: shadow")
        print(f"Observation source: {args.observation_source}")
        print(f"Runner: {args.runner_url}")
        print(f"FPS target: {args.fps}")
        print(f"Max steps: {args.max_steps}")

    try:
        result = run_shadow_mode(config)
    except CoreAIPolicyError as e:
        print(f"\n✗ Shadow failed: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        report_path = Path(args.output_dir) / "shadow_report.json"
        if report_path.exists():
            print(f"Failure report: {report_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(result.report, indent=2))
        return 0 if result.ok else 1

    m = result.report.get("metrics", {})
    print()
    print(f"✓ Policy loaded")
    print(f"✓ Runner supports action")
    print(f"✓ Observation source opened")
    print(f"✓ {m.get('observations_read', 0)} observations read")
    print(f"✓ {m.get('actions_generated', 0)} actions generated")
    print(f"✓ {m.get('actions_blocked', 0)} actions blocked")
    print(f"✓ No robot commands sent")
    print()
    print("Files:")
    print(f"  actions:         {result.actions_path}")
    print(f"  blocked actions: {result.blocked_actions_path}")
    print(f"  observations:    {result.observations_path}")
    print(f"  trace:           {result.trace_path}")
    print(f"  report:          {result.report_path}")
    print("=" * 50)
    print("Shadow run completed successfully.")

    return 0


# MARK: - sim (v0.8 — simulator-only action egress)

def cmd_sim(args: argparse.Namespace) -> int:
    """Run simulator-only sim mode: observe, generate actions, egress to simulator only."""
    from .errors import CoreAIPolicyError

    # Parse state-vector (comma-separated floats) if provided.
    state_vector = None
    if args.state_vector:
        state_vector = [float(v.strip()) for v in args.state_vector.split(",")]

    config = SimConfig(
        policy_path=args.policy_path,
        runner_url=args.runner_url,
        output_dir=Path(args.output_dir),
        env_type=args.env_type,
        robot_type=args.robot_type,
        task=args.task,
        state_vector=state_vector,
        image_key=args.image_key,
        env_config=Path(args.env_config) if args.env_config else None,
        env_render=getattr(args, "env_render", False),
        env_record_video=getattr(args, "env_record_video", False),
        env_video_dir=Path(args.env_video_dir) if args.env_video_dir else None,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        seed=args.seed,
        fps=args.fps,
        strict_observation_keys=args.strict,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
        live=getattr(args, "live", False),
        live_every=getattr(args, "live_every", 1),
        confirm_sim_egress=getattr(args, "confirm_sim_egress", False),
    )

    if not args.json:
        print(f"lerobot-coreai sim")
        print("=" * 50)
        print(f"Policy: {args.policy_path}")
        print(f"Mode: sim")
        print(f"Environment: {args.env_type}")
        print(f"Runner: {args.runner_url}")
        print(f"Episodes: {args.episodes}")
        print(f"Max steps/episode: {args.max_steps_per_episode}")
        print(f"FPS target: {args.fps}")

    try:
        result = run_sim_mode(config)
    except CoreAIPolicyError as e:
        print(f"\n✗ Sim failed: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        report_path = Path(args.output_dir) / "sim_report.json"
        if report_path.exists():
            print(f"Failure report: {report_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(result.report, indent=2))
        return 0 if result.ok else 1

    m = result.report.get("metrics", {})
    print()
    print(f"✓ Policy loaded")
    print(f"✓ Runner supports action")
    print(f"✓ Environment built ({args.env_type})")
    print(f"✓ {m.get('episodes_completed', 0)} episodes completed")
    print(f"✓ {m.get('steps_completed', 0)} steps completed")
    print(f"✓ {m.get('actions_generated', 0)} actions generated")
    print(f"✓ {m.get('actions_sent_to_simulator', 0)} actions sent to simulator")
    print(f"✓ 0 actions sent to robot")
    print(f"✓ No robot commands sent")
    print()
    print("Files:")
    print(f"  actions:      {result.actions_path}")
    print(f"  episodes:     {result.episodes_path}")
    print(f"  observations: {result.observations_path}")
    print(f"  trace:        {result.trace_path}")
    print(f"  report:       {result.report_path}")
    print("=" * 50)
    print("Sim run completed successfully.")

    return 0


# MARK: - compare (v0.5 — PyTorch vs CoreAI action parity)

def cmd_compare(args: argparse.Namespace) -> int:
    """Compare PyTorch vs CoreAI action parity on LeRobotDataset."""
    from .errors import CoreAIPolicyError

    episodes = None
    if args.episodes:
        episodes = [int(e.strip()) for e in args.episodes.split(",")]

    output_dir = args.output_dir or f"runs/{args.coreai_policy_path.split('/')[-1]}-compare"

    config = CompareConfig(
        torch_policy_path=args.torch_policy_path,
        coreai_policy_path=args.coreai_policy_path,
        dataset_repo_id=args.dataset_repo_id,
        runner_url=args.runner_url,
        output_dir=Path(output_dir),
        robot_type=args.robot_type,
        torch_policy_type=args.torch_policy_type,
        max_frames=args.max_frames,
        start_index=args.start_index,
        stride=args.stride,
        episodes=episodes,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        dataset_revision=args.dataset_revision,
        download_videos=args.download_videos,
        video_backend=args.video_backend,
        strict_observation_keys=args.strict,
        fail_fast=args.fail_fast,
        overwrite=args.overwrite,
        tolerance_cosine=args.tolerance_cosine,
        tolerance_max_mae=args.tolerance_max_mae,
        tolerance_mean_mae=args.tolerance_mean_mae,
        save_actions=args.save_actions,
        reset_each_frame=args.reset_each_frame,
    )

    if not args.json:
        print(f"lerobot-coreai compare")
        print("=" * 50)
        print(f"PyTorch:  {args.torch_policy_path}")
        print(f"CoreAI:   {args.coreai_policy_path}")
        print(f"Dataset:  {args.dataset_repo_id}")
        print(f"Runner:   {args.runner_url}")
        print(f"Max frames: {args.max_frames}")

    try:
        result = run_lerobot_policy_compare(config)
    except CoreAIPolicyError as e:
        print(f"\n✗ Compare failed: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(result.report, indent=2))
        return 0 if result.ok else 1

    m = result.report.get("metrics", {})
    c = result.report.get("claims", {})
    print()
    print(f"✓ Compare completed")
    print(f"  frames compared: {m.get('frames_compared', 0)}")
    print(f"  frames passed:   {m.get('frames_passed', 0)}")
    print(f"  frames failed:   {m.get('frames_failed', 0)}")
    if m.get("min_cosine_similarity") is not None:
        print(f"  min cosine:      {m.get('min_cosine_similarity', 0):.10f}")
        print(f"  mean cosine:     {m.get('mean_cosine_similarity', 0):.10f}")
        print(f"  max MAE:         {m.get('max_absolute_error', 0):.10f}")
    print(f"  numeric fidelity: {'YES' if c.get('proves_numeric_action_fidelity') else 'NO'}")
    print(f"  No robot commands sent")
    print()
    print("Files:")
    print(f"  actions: {result.actions_path}")
    print(f"  trace:   {result.trace_path}")
    print(f"  report:  {result.report_path}")
    print("=" * 50)
    if c.get("proves_numeric_action_fidelity"):
        print("Numeric action parity PROVEN on {} frames.".format(m.get('frames_compared', 0)))
    else:
        print("Numeric action parity NOT proven.")

    return 0 if result.ok else 1


# MARK: - export (v0.6 — export/verify/package pipeline)

def cmd_export(args: argparse.Namespace) -> int:
    """Export/verify/package a LeRobot policy as a CoreAI artifact."""
    from .errors import CoreAIPolicyError

    config = ExportConfig(
        torch_policy_path=args.torch_policy_path,
        output_dir=Path(args.output_dir),
        policy_type=args.policy_type,
        robot_type=args.robot_type,
        dataset_repo_id=args.dataset_repo_id,
        runner_url=args.runner_url,
        model_id=args.model_id,
        output_repo_id=args.output_repo_id,
        artifact_name=args.artifact_name,
        fabric_config=Path(args.fabric_config) if args.fabric_config else None,
        fabric_profile=args.fabric_profile,
        fabric_target=args.fabric_target,
        skip_fabric=args.skip_fabric,
        existing_artifact=Path(args.existing_artifact) if args.existing_artifact else None,
        verify_runner=args.verify_runner,
        dry_run_fixture=Path(args.dry_run_fixture) if args.dry_run_fixture else None,
        eval_max_frames=args.eval_max_frames,
        compare_max_frames=args.compare_max_frames,
        compare_tolerance_cosine=args.compare_tolerance_cosine,
        compare_tolerance_max_mae=args.compare_tolerance_max_mae,
        compare_tolerance_mean_mae=args.compare_tolerance_mean_mae,
        publish_ready=args.publish_ready,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
    )

    if not args.json:
        print(f"lerobot-coreai export")
        print("=" * 50)
        print(f"Source policy: {args.torch_policy_path}")
        print(f"Output dir:    {args.output_dir}")
        print(f"Fabric:        {'skipped' if args.skip_fabric else 'enabled'}")
        print(f"Runner verify: {'yes' if args.verify_runner else 'no'}")
        print(f"Dry-run:       {'yes' if args.dry_run_fixture else 'no'}")
        print(f"Eval:          {args.eval_max_frames} frames" if args.eval_max_frames > 0 else "Eval:          skipped")
        print(f"Compare:       {args.compare_max_frames} frames" if args.compare_max_frames > 0 else "Compare:       skipped")

    try:
        result = run_coreai_export_pipeline(config)
    except CoreAIPolicyError as e:
        print(f"\n✗ Export failed: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        print("No robot commands were sent.", file=sys.stderr)
        return 1

    if args.json:
        import json
        print(json.dumps(result.report, indent=2))
        return 0 if result.ok else 1

    v = result.report.get("verification", {})
    c = result.report.get("claims", {})
    print()
    if result.artifact_path:
        print(f"✓ Artifact exported")
    print(f"✓ Manifest valid: {v.get('manifest_valid', False)}")
    if v.get("runner_checked"):
        print(f"✓ Runner supports action: {v.get('runner_ok', False)}")
    if v.get("dry_run", {}).get("ran"):
        print(f"✓ Dry-run: {'passed' if v['dry_run'].get('ok') else 'failed'}")
    if v.get("eval", {}).get("ran"):
        print(f"✓ Eval: {'passed' if v['eval'].get('ok') else 'failed'}")
    if v.get("compare", {}).get("ran"):
        print(f"✓ Compare: {'passed' if v['compare'].get('ok') else 'failed'}")
    if c.get("publish_ready"):
        print(f"✓ Publish folder ready")
    print(f"✓ No robot commands sent")
    print()
    print("Files:")
    print(f"  report: {result.report_path}")
    print(f"  trace:  {result.trace_path}")
    if result.manifest_path:
        print(f"  manifest: {result.manifest_path}")
    print("=" * 50)
    if c.get("proves_numeric_action_fidelity"):
        print("Numeric action fidelity PROVEN.")
    else:
        print("Export completed. Numeric fidelity not proven (no compare or compare failed).")

    return 0 if result.ok else 1
