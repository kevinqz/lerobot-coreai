# manifest.py — parse and validate lerobot-coreai.json (the compatibility manifest, spec §14).
#
# Every HF artifact produced for a LeRobot-derived CoreAI policy includes lerobot-coreai.json.
# This module downloads it, validates it against the JSON Schema, and exposes typed access.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import requests

from .errors import ManifestError

# --- Constants ---

SCHEMA_VERSION = "lerobot-coreai.v0"
MANIFEST_FILENAME = "lerobot-coreai.json"
HF_RAW_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"

# The schema is bundled in the package for offline validation.
_SCHEMA_PATH = Path(__file__).parent.parent.parent / "schemas" / "lerobot-coreai.schema.json"


def _load_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for lerobot-coreai.json."""
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


@dataclass
class FeatureSpec:
    """A single observation or action feature."""
    dtype: str
    shape: list[int] | None = None
    required: bool = True


@dataclass
class GraphSpec:
    """A runtime graph in the artifact."""
    name: str
    role: str


@dataclass
class LeRobotCoreAIManifest:
    """Typed view of a lerobot-coreai.json manifest.

    Use ``load_manifest(repo_id)`` to download and validate from Hugging Face,
    or ``LeRobotCoreAIManifest.from_dict(data)`` to parse from an already-loaded dict.
    """
    schema_version: str
    runtime: str
    # Framework
    framework_name: str
    framework_version: str
    framework_commit: str | None
    # Policy
    policy_repo_id: str
    policy_source_repo_id: str
    policy_type: str
    policy_class: str | None
    policy_config_class: str | None
    # Robot
    robot_type: str
    robot_action_representation: str | None
    robot_fps: int | None
    # Features
    observation_features: dict[str, FeatureSpec]
    action_features: dict[str, FeatureSpec]
    # Normalization
    normalization_format: str
    normalization_path: str
    normalization_sha256: str | None
    # CoreAI
    artifact_format: str
    runner: str
    graphs: list[GraphSpec]
    host_loop_required: bool
    host_loop_type: str | None
    host_loop_solver: str | None
    host_loop_num_steps: int | None
    # Evaluation
    evaluation_metric: str | None
    evaluation_status: str
    evaluation_n_obs: int | None
    evaluation_min_chunk_cosine: float | None
    evaluation_max_action_mae: float | None
    proves_numeric_fidelity: bool
    proves_task_success: bool
    proves_robot_safety: bool
    # Safety
    default_mode: str
    real_actuation_requires_confirmation: bool
    # Raw
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeRobotCoreAIManifest":
        """Parse and validate a manifest dict. Raises ManifestError on invalid data."""
        try:
            jsonschema.validate(instance=data, schema=_load_schema())
        except jsonschema.ValidationError as e:
            raise ManifestError(
                f"lerobot-coreai.json failed schema validation: {e.message}",
            ) from e
        except FileNotFoundError as e:
            raise ManifestError(f"Schema file not found: {_SCHEMA_PATH}") from e

        fw = data["framework"]
        pol = data["policy"]
        rob = data["robot"]
        feats = data["features"]
        norm = data["normalization"]
        coreai = data["coreai"]
        eval_block = data["evaluation"]
        safety = data["safety"]

        obs_features = {
            name: FeatureSpec(
                dtype=f["dtype"],
                shape=f.get("shape"),
                required=f.get("required", True),
            )
            for name, f in feats["observation"].items()
        }
        act_features = {
            name: FeatureSpec(
                dtype=f["dtype"],
                shape=f.get("shape"),
                required=f.get("required", True),
            )
            for name, f in feats["action"].items()
        }

        graphs = [GraphSpec(name=g["name"], role=g["role"]) for g in coreai.get("graphs", [])]

        host_loop = coreai.get("host_loop")

        return cls(
            schema_version=data["schema_version"],
            runtime=data["runtime"],
            framework_name=fw["name"],
            framework_version=fw["version"],
            framework_commit=fw.get("commit"),
            policy_repo_id=pol["repo_id"],
            policy_source_repo_id=pol["source_repo_id"],
            policy_type=pol["type"],
            policy_class=pol.get("class"),
            policy_config_class=pol.get("config_class"),
            robot_type=rob["type"],
            robot_action_representation=rob.get("action_representation"),
            robot_fps=rob.get("fps"),
            observation_features=obs_features,
            action_features=act_features,
            normalization_format=norm["format"],
            normalization_path=norm["path"],
            normalization_sha256=norm.get("sha256"),
            artifact_format=coreai["artifact_format"],
            runner=coreai["runner"],
            graphs=graphs,
            host_loop_required=coreai.get("host_loop_required", False),
            host_loop_type=host_loop.get("type") if host_loop else None,
            host_loop_solver=host_loop.get("solver") if host_loop else None,
            host_loop_num_steps=host_loop.get("num_steps") if host_loop else None,
            evaluation_metric=eval_block.get("metric"),
            evaluation_status=eval_block["status"],
            evaluation_n_obs=eval_block.get("n_obs"),
            evaluation_min_chunk_cosine=eval_block.get("min_chunk_cosine"),
            evaluation_max_action_mae=eval_block.get("max_action_mae"),
            proves_numeric_fidelity=eval_block.get("proves_numeric_fidelity", False),
            proves_task_success=eval_block.get("proves_task_success", False),
            proves_robot_safety=eval_block.get("proves_robot_safety", False),
            default_mode=safety["default_mode"],
            real_actuation_requires_confirmation=safety.get("real_actuation_requires_confirmation", True),
            raw=data,
        )

    @property
    def parity_passed(self) -> bool:
        """True if action parity was verified and passed."""
        return self.evaluation_status == "passed"

    @property
    def lerobot_version_supported(self) -> str:
        """The LeRobot version this artifact was exported against."""
        return self.framework_version


def load_manifest(repo_id: str, *, revision: str = "main") -> LeRobotCoreAIManifest:
    """Download and validate lerobot-coreai.json from a Hugging Face artifact repo.

    Args:
        repo_id: HF repo id (e.g. 'kevinqz/EVO1-SO100-CoreAI').
        revision: HF revision (default 'main').

    Returns:
        Parsed and validated manifest.

    Raises:
        ManifestError: If the file is missing or fails validation.
        DownloadError: If the download fails.
    """
    from .errors import DownloadError

    url = HF_RAW_BASE.format(repo=repo_id, filename=MANIFEST_FILENAME)
    # Override revision in the URL
    url = url.replace("/resolve/main/", f"/resolve/{revision}/")

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as e:
        raise DownloadError(f"Failed to fetch {MANIFEST_FILENAME} from {repo_id}: {e}") from e

    if resp.status_code == 404:
        raise ManifestError(
            f"{MANIFEST_FILENAME} not found in {repo_id}. "
            f"This repo may not be a CoreAI-backed LeRobot artifact.",
            repo_id=repo_id,
        )
    if resp.status_code != 200:
        raise DownloadError(
            f"HTTP {resp.status_code} fetching {MANIFEST_FILENAME} from {repo_id}"
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise ManifestError(f"{MANIFEST_FILENAME} in {repo_id} is not valid JSON: {e}") from e

    return LeRobotCoreAIManifest.from_dict(data)
