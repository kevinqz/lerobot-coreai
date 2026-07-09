# conftest.py — shared pytest fixtures.

import json
from pathlib import Path

import pytest


# A valid manifest matching the spec §14.1 example (EVO1-SO100-CoreAI).
VALID_MANIFEST = {
    "schema_version": "lerobot-coreai.v0",
    "runtime": "coreai",
    "framework": {
        "name": "lerobot",
        "version": "0.6.0",
        "commit": None,
    },
    "policy": {
        "repo_id": "kevinqz/EVO1-SO100-CoreAI",
        "source_repo_id": "lerobot/evo1_so100",
        "type": "evo1",
        "class": None,
        "config_class": None,
    },
    "robot": {
        "type": "so100",
        "action_representation": "joint_position_delta",
        "fps": 30,
    },
    "features": {
        "observation": {
            "observation.images.wrist": {
                "dtype": "image",
                "shape": [3, 224, 224],
                "required": True,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [7],
                "required": True,
            },
            "task": {
                "dtype": "string",
                "required": False,
            },
        },
        "action": {
            "action": {
                "dtype": "float32",
                "shape": [16, 7],
            },
        },
    },
    "normalization": {
        "format": "lerobot",
        "path": "norm_stats.json",
        "sha256": None,
    },
    "coreai": {
        "artifact_format": "aimodel",
        "runner": "coreai-runner",
        "graphs": [
            {"name": "action_denoise_step", "role": "denoise_step"},
        ],
        "host_loop_required": True,
        "host_loop": {
            "type": "flow_matching",
            "solver": "euler",
            "num_steps": 10,
        },
    },
    "evaluation": {
        "metric": "action_parity",
        "status": "passed",
        "n_obs": 8,
        "min_chunk_cosine": 0.9999999999999983,
        "max_action_mae": None,
        "max_relative_action_mae": None,
        "proves_numeric_fidelity": True,
        "proves_task_success": False,
        "proves_robot_safety": False,
    },
    "safety": {
        "default_mode": "dry_run",
        "real_actuation_requires_confirmation": True,
    },
}


@pytest.fixture
def valid_manifest_dict():
    """A valid lerobot-coreai.json manifest dict."""
    return json.loads(json.dumps(VALID_MANIFEST))  # deep copy


@pytest.fixture
def valid_manifest_path(tmp_path, valid_manifest_dict):
    """A valid manifest written to a temp file."""
    p = tmp_path / "lerobot-coreai.json"
    p.write_text(json.dumps(valid_manifest_dict, indent=2))
    return p


def _sim_report_for_bundle():
    return {
        "schema_version": "lerobot-coreai.sim_report.v0",
        "lerobot_coreai_version": "0.9.3", "ok": True, "mode": "sim",
        "policy": {"path": "kevinqz/EVO1-SO100-CoreAI", "runtime": "coreai", "type": "evo1"},
        "runner": {"url": "http://127.0.0.1:8710", "reachable": True, "supports_action": True},
        "environment": {"type": "gym", "id": "PushT-v0", "episodes": 2,
                        "max_steps_per_episode": 10, "seed": 42,
                        "simulator_egress_enabled": True},
        "loop": {"fps_target": 0, "episodes_completed": 2, "steps_completed": 20},
        "metrics": {"episodes_completed": 2, "steps_completed": 20, "mean_episode_reward": 1.0},
        "episode_metrics": {"success_rate": 1.0, "mean_reward": 1.0},
        "claims": {"proves_sim_task_success": True, "proves_real_task_success": False,
                   "proves_robot_safety": False, "proves_real_world_safety": False},
        "safety": {"simulator_egress_enabled": True, "robot_egress_enabled": False,
                   "physical_actuation_possible": False, "actions_sent_to_robot": 0,
                   "action_egress": "simulator_only"},
        "safety_supervisor": {"enabled": True, "mode": "enforce",
                              "profile": "so100-sim-default", "actions_supervised": 20,
                              "actions_blocked": 0, "passed": True},
        "files": {"report": "sim_report.json"}, "errors": [],
    }


def _safety_summary(passed=True, blocked=0):
    return {
        "schema_version": "lerobot-coreai.safety_summary.v0", "profile": "so100-sim-default",
        "mode": "enforce", "actions_supervised": 20, "actions_allowed": 20 - blocked,
        "actions_blocked": blocked, "actions_modified": 0, "critical_failures": 0,
        "would_block_actions": 0, "critical_findings": 0, "top_reasons": {}, "passed": passed,
        "claims": {"proves_software_supervision": True, "proves_physical_safety": False,
                   "proves_real_world_safety": False, "proves_real_task_success": False},
    }


@pytest.fixture
def sim_evidence_bundle(tmp_path):
    """Build a complete, verifiable sim evidence bundle for approval tests.

    Returns a factory: make(passed_quality=True, passed_regression=True,
    with_regression=True) -> bundle_dir.
    """
    from lerobot_coreai.sim_bundle import SimBundleConfig, package_sim_run

    def _make(passed_quality=True, passed_regression=True, with_regression=True,
              with_calibration=True, name="bundle"):
        run = tmp_path / f"run_{name}"
        run.mkdir(parents=True, exist_ok=True)
        (run / "sim_report.json").write_text(json.dumps(_sim_report_for_bundle()))
        (run / "safety_summary.json").write_text(json.dumps(_safety_summary()))
        (run / "actions.jsonl").write_text('{"step":0,"action":[[0.0]]}\n')
        (run / "safety_quality_report.json").write_text(json.dumps({
            "schema_version": "lerobot-coreai.safety_quality_report.v0",
            "lerobot_coreai_version": "0.9.3", "passed": passed_quality,
            "summary": {"actions_blocked": 0}, "checks": [],
            "claims": {"proves_software_safety_quality": True,
                       "proves_physical_safety": False, "proves_real_world_safety": False,
                       "proves_real_task_success": False},
        }))
        if with_regression:
            (run / "safety_regression_report.json").write_text(json.dumps({
                "schema_version": "lerobot-coreai.safety_regression_report.v0",
                "passed": passed_regression, "baseline": {}, "candidate": {}, "deltas": {},
                "checks": [],
                "claims": {"proves_no_safety_regression_on_compared_artifacts": passed_regression,
                           "proves_physical_safety": False, "proves_real_world_safety": False},
            }))
        if with_calibration:
            (run / "calibrated_profile.json").write_text(json.dumps({
                "schema_version": "lerobot-coreai.safety_profile.v0",
                "name": "so100-calibrated", "profile_type": "software_bounds",
                "mode": "fail_closed", "max_abs_action": 1.0,
                "limitations": ["Does not prove physical safety."],
            }))
        bundle = tmp_path / f"bundle_{name}"
        package_sim_run(SimBundleConfig(run_dir=run, output_dir=bundle, overwrite=True))
        return bundle

    return _make
