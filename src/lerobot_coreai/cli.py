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
    p_export = sub.add_parser("export", help="Export a LeRobot policy to CoreAI (v0.4)")
    p_export.add_argument("--policy.path", dest="policy_path", required=True)
    p_export.add_argument("--output.repo_id", dest="output_repo_id", required=True)
    p_export.set_defaults(func=cmd_not_implemented)

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
        f"'{args.command}' is not implemented in v0.4. "
        f"Available commands: inspect, doctor, list, predict, rollout --mode dry_run, eval.",
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
        print(f"  min cosine:      {m['min_cosine_similarity']:.10f}")
        print(f"  mean cosine:     {m['mean_cosine_similarity']:.10f}")
        print(f"  max MAE:         {m['max_absolute_error']:.10f}")
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
