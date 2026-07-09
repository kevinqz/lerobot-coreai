# sim_bundle.py — package a sim run into a reproducibility bundle (v0.8.4).
#
# A bundle is a self-contained, auditable artifact: it copies the source run's
# report/traces/actions/CSVs, derives policy/environment/runner metadata, writes
# a manifest with SHA256 checksums, and includes README + reproducibility notes.
#
# The packager refuses to bundle a report that violates no-robot-egress
# invariants. A bundle never proves real-world task success or physical safety.

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError


# MARK: - Config / Result dataclasses

@dataclass
class SimBundleConfig:
    """Configuration for packaging a sim run into a bundle."""

    run_dir: Path
    output_dir: Path
    overwrite: bool = False
    include_actions: bool = True
    include_traces: bool = True
    include_observations_jsonl: bool = True
    include_observations_dir: bool = False
    include_csv: bool = True
    include_summary: bool = True
    include_failure_taxonomy: bool = True
    redact_runner_url: bool = False
    redact_local_paths: bool = True
    created_by: str = "lerobot-coreai"


@dataclass
class SimBundleResult:
    """Result of packaging a sim run."""

    ok: bool
    output_dir: Path
    manifest_path: Path
    checksums_path: Path
    files_copied: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class SimBundleVerificationResult:
    """Result of verifying a sim bundle."""

    ok: bool
    bundle_dir: Path
    files_checked: int = 0
    checksum_failures: list[str] = field(default_factory=list)
    invariant_failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# MARK: - Checksum helpers

def sha256_file(path: Path) -> str:
    """Compute the SHA256 hash of a file, returned as 'sha256:<hex>'."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def build_checksums(output_dir: Path) -> dict[str, Any]:
    """Compute SHA256 for all files in the bundle (excluding checksums.json itself)."""
    output_dir = Path(output_dir)
    files: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel == "checksums.json":
            continue
        files[rel] = sha256_file(path)
    return {"algorithm": "sha256", "files": files}


def verify_bundle_checksums(bundle_dir: Path) -> dict[str, Any]:
    """Verify all checksums in a bundle. Returns {ok, checked, failures}."""
    bundle_dir = Path(bundle_dir)
    checksums_path = bundle_dir / "checksums.json"
    if not checksums_path.is_file():
        return {"ok": False, "checked": 0, "failures": ["checksums.json missing"]}
    stored = json.loads(checksums_path.read_text())
    files = stored.get("files", {})
    failures: list[str] = []
    for rel, expected in files.items():
        path = bundle_dir / rel
        if not path.is_file():
            failures.append(f"{rel}: missing")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"{rel}: checksum mismatch")
    return {"ok": len(failures) == 0, "checked": len(files), "failures": failures}


# MARK: - No-robot-egress invariant validation

def _validate_source_report(report: dict[str, Any]) -> list[str]:
    """Validate no-robot-egress and no-overclaim invariants on the source report.

    Returns a list of human-readable failure strings (empty if the report is a
    valid, honest simulator-only report). Covers both the safety block (no robot
    egress) and the claims block (no real-world/safety overclaim). A report that
    fails any of these must never be packaged.
    """
    failures: list[str] = []
    if report.get("mode") != "sim":
        failures.append("mode is not 'sim'")

    safety = report.get("safety", {}) or {}
    if safety.get("robot_egress_enabled") is not False:
        failures.append("safety.robot_egress_enabled is not false")
    if safety.get("actions_sent_to_robot") != 0:
        failures.append("safety.actions_sent_to_robot is not 0")
    if safety.get("action_egress") != "simulator_only":
        failures.append("safety.action_egress is not 'simulator_only'")
    if safety.get("physical_actuation_possible") is not False:
        failures.append("safety.physical_actuation_possible is not false")

    # No-overclaim: a sim report must not claim real-world success or safety.
    if report.get("claims") is None:
        failures.append("claims block missing")
    else:
        claims = report["claims"]
        if claims.get("proves_real_task_success") is not False:
            failures.append("claims.proves_real_task_success is not false")
        if claims.get("proves_robot_safety") is not False:
            failures.append("claims.proves_robot_safety is not false")
        if claims.get("proves_real_world_safety") is not False:
            failures.append("claims.proves_real_world_safety is not false")
    return failures


# MARK: - File copy policy

def _copy_if_present(src: Path, dst: Path, warnings: list[str]) -> bool:
    """Copy src to dst if it is a file; warn if missing or not a file.

    Returns True only when a regular file was copied.
    """
    if not src.exists():
        warnings.append(f"optional file not found: {src.name}")
        return False
    if not src.is_file():
        warnings.append(f"optional path is not a file: {src.name}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_dir_if_present(src: Path, dst: Path, warnings: list[str]) -> bool:
    """Copy a directory tree if src exists; warn if missing."""
    if not src.is_dir():
        warnings.append(f"optional directory not found: {src.name}")
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


# MARK: - Manifest / metadata builders

def _redact_local_path(path: Path, redact: bool) -> str:
    """Redact an absolute local path to its basename (keeps the run identifier,
    drops the home/user prefix). Relative paths are left untouched."""
    path = Path(path)
    if redact and path.is_absolute():
        return path.name or "<redacted-local-path>"
    return str(path)

def _build_policy_metadata(report: dict[str, Any]) -> dict[str, Any]:
    policy = report.get("policy", {}) or {}
    return {
        "path": policy.get("path"),
        "runtime": policy.get("runtime", "coreai"),
        "policy_type": policy.get("type"),
        "robot_type": policy.get("robot_type"),
        "source_repo_id": policy.get("source_repo_id"),
        "model_id": policy.get("model_id"),
    }


def _build_environment_metadata(report: dict[str, Any]) -> dict[str, Any]:
    env = report.get("environment", {}) or {}
    loop = report.get("loop", {}) or {}
    return {
        "type": env.get("type"),
        "id": env.get("id"),
        "episodes": env.get("episodes"),
        "max_steps_per_episode": env.get("max_steps_per_episode"),
        "seed": env.get("seed"),
        "fps_target": loop.get("fps_target"),
    }


def _build_runner_metadata(report: dict[str, Any], redact: bool) -> dict[str, Any]:
    runner = report.get("runner", {}) or {}
    url = runner.get("url")
    return {
        "url": "<redacted>" if redact else url,
        "redacted": redact,
        "reachable_at_run_time": runner.get("reachable"),
        "supports_action": runner.get("supports_action"),
    }


def _build_bundle_manifest(
    *,
    config: SimBundleConfig,
    report: dict[str, Any],
    files_copied: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Build the bundle_manifest.json dict."""
    metrics = report.get("metrics", {}) or {}
    episode_m = report.get("episode_metrics", {}) or {}
    safety = report.get("safety", {}) or {}
    claims = report.get("claims", {}) or {}

    results = {
        "ok": report.get("ok"),
        "episodes_completed": metrics.get("episodes_completed"),
        "steps_completed": metrics.get("steps_completed"),
        "success_rate": episode_m.get("success_rate"),
        "mean_episode_reward": metrics.get("mean_episode_reward"),
    }
    analytics = {
        "has_episode_metrics": "episode_metrics" in report,
        "has_latency_metrics": "latency_metrics" in report,
        "has_action_metrics": "action_metrics" in report,
        "has_failure_metrics": "failure_metrics" in report,
        "has_quality": "quality" in report,
    }

    # Semantic file keys (stable public names) rather than raw filename stems.
    _key_by_name = {
        "sim_report.json": "report",
        "sim_summary.md": "summary",
        "failure_taxonomy.json": "failure_taxonomy",
        "sim_trace.jsonl": "trace",
        "actions.jsonl": "actions",
        "episodes.jsonl": "episodes",
        "observations.jsonl": "observations",
        "episode_metrics.csv": "episode_metrics_csv",
        "step_metrics.csv": "step_metrics_csv",
        "safety_report.jsonl": "safety_report",
        "safety_summary.json": "safety_summary",
        "safety_summary.md": "safety_summary_md",
    }
    files_map = {"checksums": "checksums.json"}
    for f in files_copied:
        name = Path(f).name
        files_map[_key_by_name.get(name, Path(f).stem)] = f

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "schema_version": "lerobot-coreai.sim_bundle.v0",
        "lerobot_coreai_version": __version__,
        "created_at": created_at,
        "created_by": config.created_by,
        "bundle_type": "sim_run",
        "mode": "sim",
        "source_run": {
            "run_dir": _redact_local_path(config.run_dir, config.redact_local_paths),
            "report": "source_run/sim_report.json",
        },
        "policy": _build_policy_metadata(report),
        "environment": _build_environment_metadata(report),
        "runner": _build_runner_metadata(report, config.redact_runner_url),
        "results": results,
        "analytics": analytics,
        "safety": {
            "simulator_egress_enabled": safety.get("simulator_egress_enabled"),
            "robot_egress_enabled": safety.get("robot_egress_enabled"),
            "actions_sent_to_robot": safety.get("actions_sent_to_robot"),
            "action_egress": safety.get("action_egress"),
            "physical_actuation_possible": safety.get("physical_actuation_possible"),
        },
        "claims": {
            "proves_sim_task_success": claims.get("proves_sim_task_success"),
            "proves_real_task_success": claims.get("proves_real_task_success"),
            "proves_robot_safety": claims.get("proves_robot_safety"),
            "proves_real_world_safety": claims.get("proves_real_world_safety"),
        },
        "files": files_map,
        "safety_supervisor": report.get("safety_supervisor"),
        "warnings": warnings,
    }


def _build_bundle_readme(report: dict[str, Any]) -> str:
    """Build the bundle README.md."""
    policy = report.get("policy", {}) or {}
    env = report.get("environment", {}) or {}
    metrics = report.get("metrics", {}) or {}
    episode_m = report.get("episode_metrics", {}) or {}
    safety = report.get("safety", {}) or {}

    sr = episode_m.get("success_rate")
    sr_str = f"{sr * 100:.1f}%" if isinstance(sr, (int, float)) else "n/a"

    sections = [
        "# lerobot-coreai Sim Run Bundle\n\n"
        "This bundle packages a simulator-only run produced by `lerobot-coreai`.",
        "## Summary\n\n"
        f"- Mode: sim\n"
        f"- Policy: {policy.get('path', 'n/a')}\n"
        f"- Environment: {env.get('type', 'n/a')} / {env.get('id', 'n/a')}\n"
        f"- Episodes completed: {metrics.get('episodes_completed', 'n/a')}\n"
        f"- Steps completed: {metrics.get('steps_completed', 'n/a')}\n"
        f"- Success rate: {sr_str}\n"
        f"- Mean reward: {metrics.get('mean_episode_reward', 'n/a')}",
        "## Safety\n\n"
        f"- Simulator egress enabled: {safety.get('simulator_egress_enabled', 'n/a')}\n"
        f"- Robot egress enabled: {safety.get('robot_egress_enabled', 'n/a')}\n"
        f"- Actions sent to robot: {safety.get('actions_sent_to_robot', 'n/a')}\n"
        f"- Action egress: {safety.get('action_egress', 'n/a')}",
        "## Claims\n\n"
        "This bundle may support simulator-level analysis. "
        "It does not prove real-world task success. "
        "It does not prove physical robot safety.",
        "## Reproduce\n\n"
        "See `reproducibility.md`.",
    ]
    return "\n\n".join(sections) + "\n"


def _build_reproducibility_md(report: dict[str, Any]) -> str:
    """Build the reproducibility.md."""
    policy = report.get("policy", {}) or {}
    env = report.get("environment", {}) or {}
    loop = report.get("loop", {}) or {}
    runner = report.get("runner", {}) or {}

    env_id = env.get("id") or "<env-id>"
    env_type = env.get("type") or "fake"

    lines = [
        "# Reproducibility",
        "",
        "This bundle records the configuration of a simulator-only run.",
        "",
        "## Original command shape",
        "",
        "```bash",
        "lerobot-coreai sim \\",
        f"  --policy.path {policy.get('path', '<policy>')} \\",
        f"  --env.type {env_type} \\",
    ]
    if env.get("id"):
        lines.append(f"  --env.id {env_id} \\")
    lines.extend([
        f"  --episodes {env.get('episodes', 1)} \\",
        f"  --max-steps-per-episode {env.get('max_steps_per_episode', 300)} \\",
    ])
    if env.get("seed") is not None:
        lines.append(f"  --seed {env.get('seed')} \\")
    lines.extend([
        "  --confirm-sim-egress \\",
        "  --export-csv \\",
        "  --output-dir runs/<run-dir>",
        "```",
        "",
        "## Notes",
        "",
        "Exact reproduction may require:",
        "",
        "- same policy artifact revision",
        "- same simulator version",
        "- same runner implementation",
        "- same random seed",
        "- same OS/Python dependency environment",
        "",
        "This bundle does not include model weights unless explicitly stated.",
        "This bundle does not prove real-world task success or physical robot safety.",
    ])
    return "\n".join(lines) + "\n"


# MARK: - Main packaging entry point

def package_sim_run(config: SimBundleConfig) -> SimBundleResult:
    """Package a simulator-only run into a reproducibility bundle.

    Requires run_dir/sim_report.json. Validates no-robot-egress invariants on
    the source report before packaging. Copies source files, derives metadata,
    writes manifest/checksums/README/repro docs.
    """
    run_dir = Path(config.run_dir)
    output_dir = Path(config.output_dir)
    warnings: list[str] = []
    files_copied: list[str] = []

    # Require source report.
    report_path = run_dir / "sim_report.json"
    if not report_path.is_file():
        raise CoreAIPolicyError(
            f"Cannot package sim run: sim_report.json not found in {run_dir}."
        )
    report = json.loads(report_path.read_text())

    # Validate no-robot-egress invariants.
    failures = _validate_source_report(report)
    if failures:
        raise CoreAIPolicyError(
            "Cannot package run: source sim_report violates no-robot-egress invariants: "
            + "; ".join(failures)
        )

    # Prepare output dir.
    if output_dir.exists() and any(output_dir.iterdir()):
        if not config.overwrite:
            raise CoreAIPolicyError(
                f"Bundle output directory not empty: {output_dir}. Use --overwrite to replace."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_run_dir = output_dir / "source_run"
    source_run_dir.mkdir(parents=True, exist_ok=True)

    # Copy required report.
    shutil.copy2(report_path, source_run_dir / "sim_report.json")
    files_copied.append("source_run/sim_report.json")

    # Copy optional files.
    optional_files: list[tuple[str, bool]] = [
        ("episodes.jsonl", config.include_actions),
        ("actions.jsonl", config.include_actions),
        ("observations.jsonl", config.include_observations_jsonl),
        ("sim_trace.jsonl", config.include_traces),
        ("sim_summary.md", config.include_summary),
        ("failure_taxonomy.json", config.include_failure_taxonomy),
        ("episode_metrics.csv", config.include_csv),
        ("step_metrics.csv", config.include_csv),
        # v0.9.0 safety supervisor artifacts (small; always included if present).
        ("safety_report.jsonl", True),
        ("safety_summary.json", True),
        ("safety_summary.md", True),
    ]
    for fname, enabled in optional_files:
        if not enabled:
            continue
        if _copy_if_present(run_dir / fname, source_run_dir / fname, warnings):
            files_copied.append(f"source_run/{fname}")

    # Optional observations dir.
    if config.include_observations_dir:
        obs_dir = run_dir / "observations"
        if obs_dir.is_dir():
            shutil.copytree(obs_dir, source_run_dir / "observations", dirs_exist_ok=True)
            files_copied.append("source_run/observations/")
        else:
            warnings.append("optional directory not found: observations/")

    # Write metadata files.
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    policy_meta = _build_policy_metadata(report)
    (output_dir / "policy.json").write_text(json.dumps(policy_meta, indent=2) + "\n")
    env_meta = _build_environment_metadata(report)
    (output_dir / "environment.json").write_text(json.dumps(env_meta, indent=2) + "\n")
    runner_meta = _build_runner_metadata(report, config.redact_runner_url)
    (output_dir / "runner.json").write_text(json.dumps(runner_meta, indent=2) + "\n")

    # Write manifest, README, repro.
    manifest = _build_bundle_manifest(
        config=config, report=report, files_copied=files_copied, warnings=warnings,
    )
    # Defence-in-depth: refuse to write a manifest that fails its own schema.
    # This should never fire for a valid source report; it guards against an
    # internal bug silently producing a non-conformant (e.g. overclaiming)
    # manifest.
    try:
        import jsonschema
        jsonschema.validate(manifest, _load_bundle_schema())
    except Exception as e:  # jsonschema.ValidationError or import failure
        message = getattr(e, "message", str(e))
        raise CoreAIPolicyError(
            f"internal error: generated bundle manifest is invalid: {message}"
        )
    manifest_path = output_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    (output_dir / "README.md").write_text(_build_bundle_readme(report))
    (output_dir / "reproducibility.md").write_text(_build_reproducibility_md(report))

    # Write metadata files.json + package_info.json.
    (metadata_dir / "files.json").write_text(json.dumps({"copied": files_copied}, indent=2) + "\n")
    (metadata_dir / "package_info.json").write_text(json.dumps({
        "lerobot_coreai_version": __version__,
        "created_by": config.created_by,
        "redact_runner_url": config.redact_runner_url,
        "redact_local_paths": config.redact_local_paths,
    }, indent=2) + "\n")

    # Checksums last (after all files written).
    checksums = build_checksums(output_dir)
    checksums_path = output_dir / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2) + "\n")

    return SimBundleResult(
        ok=True,
        output_dir=output_dir,
        manifest_path=manifest_path,
        checksums_path=checksums_path,
        files_copied=files_copied,
        warnings=warnings,
    )


# MARK: - Verification entry point

def _load_bundle_schema() -> dict[str, Any]:
    from importlib.resources import files
    return json.loads(
        files("lerobot_coreai.schemas").joinpath("sim-bundle.schema.json").read_text()
    )


def verify_sim_bundle(bundle_dir: Path) -> SimBundleVerificationResult:
    """Verify a sim bundle end-to-end.

    Checks, in order:
    1. bundle_manifest.json is present
    2. the manifest validates against sim-bundle.schema.json (pins every
       safety/claims invariant)
    3. explicit invariant checks (readable failures, defence-in-depth)
    4. every checksum in checksums.json matches its file
    5. source_run/sim_report.json is present and passes the source-report
       no-robot-egress / no-overclaim invariants
    """
    bundle_dir = Path(bundle_dir)
    result = SimBundleVerificationResult(ok=True, bundle_dir=bundle_dir)
    warnings: list[str] = []

    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        return SimBundleVerificationResult(
            ok=False, bundle_dir=bundle_dir,
            invariant_failures=["bundle_manifest.json missing"],
        )

    manifest = json.loads(manifest_path.read_text())
    invariant_failures: list[str] = []

    # (2) Full schema validation — the strongest single check.
    try:
        import jsonschema
        jsonschema.validate(manifest, _load_bundle_schema())
    except Exception as e:  # jsonschema.ValidationError or import failure
        message = getattr(e, "message", str(e))
        invariant_failures.append(f"manifest schema validation failed: {message}")

    # (3) Explicit invariant checks (readable, defence-in-depth).
    safety = manifest.get("safety", {}) or {}
    claims = manifest.get("claims", {}) or {}
    if manifest.get("schema_version") != "lerobot-coreai.sim_bundle.v0":
        invariant_failures.append("manifest schema_version is invalid")
    if manifest.get("bundle_type") != "sim_run":
        invariant_failures.append("manifest bundle_type is not sim_run")
    if manifest.get("mode") != "sim":
        invariant_failures.append("manifest mode is not sim")
    if safety.get("robot_egress_enabled") is not False:
        invariant_failures.append("manifest safety.robot_egress_enabled is not false")
    if safety.get("actions_sent_to_robot") != 0:
        invariant_failures.append("manifest safety.actions_sent_to_robot is not 0")
    if safety.get("action_egress") != "simulator_only":
        invariant_failures.append("manifest safety.action_egress is not simulator_only")
    if safety.get("physical_actuation_possible") is not False:
        invariant_failures.append("manifest safety.physical_actuation_possible is not false")
    if claims.get("proves_real_task_success") is not False:
        invariant_failures.append("manifest claims.proves_real_task_success is not false")
    if claims.get("proves_robot_safety") is not False:
        invariant_failures.append("manifest claims.proves_robot_safety is not false")
    if claims.get("proves_real_world_safety") is not False:
        invariant_failures.append("manifest claims.proves_real_world_safety is not false")

    # (4) Checksum verification.
    chk = verify_bundle_checksums(bundle_dir)
    result.files_checked = chk["checked"]
    result.checksum_failures = chk["failures"]
    if chk["failures"]:
        warnings.append(f"{len(chk['failures'])} checksum failure(s)")

    # (5) Source report must exist and preserve the invariants too.
    source_report_path = bundle_dir / "source_run" / "sim_report.json"
    if not source_report_path.is_file():
        invariant_failures.append("source_run/sim_report.json missing")
    else:
        try:
            source_report = json.loads(source_report_path.read_text())
            invariant_failures.extend(
                f"source report: {f}" for f in _validate_source_report(source_report)
            )
        except json.JSONDecodeError as e:
            invariant_failures.append(f"source report unreadable: {e}")

    # (6) If a safety summary is bundled, it must not overclaim.
    safety_summary_path = bundle_dir / "source_run" / "safety_summary.json"
    if safety_summary_path.is_file():
        try:
            ss = json.loads(safety_summary_path.read_text())
            claims = ss.get("claims", {}) or {}
            for key in ("proves_physical_safety", "proves_real_world_safety",
                        "proves_real_task_success"):
                if claims.get(key) is not False:
                    invariant_failures.append(f"safety summary claims.{key} is not false")
        except json.JSONDecodeError as e:
            invariant_failures.append(f"safety summary unreadable: {e}")

    result.invariant_failures = invariant_failures
    result.warnings = warnings
    result.ok = len(result.checksum_failures) == 0 and len(invariant_failures) == 0
    return result
