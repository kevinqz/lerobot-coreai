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
