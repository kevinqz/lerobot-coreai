# manifest.py — parse and validate lerobot-coreai.json (the compatibility manifest, spec §14).
#
# Every HF artifact produced for a LeRobot-derived CoreAI policy includes lerobot-coreai.json.
# This module downloads it, validates it against the JSON Schema, and exposes typed access.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import jsonschema
import httpx

from .errors import ManifestError

# --- Constants ---

SCHEMA_VERSION = "lerobot-coreai.v0"
MANIFEST_FILENAME = "lerobot-coreai.json"
HF_RAW_BASE = "https://huggingface.co/{repo}/resolve/main/{filename}"


def _load_schema() -> dict[str, Any]:
    """Load the bundled JSON Schema for lerobot-coreai.json via importlib.resources.

    The schema lives at src/lerobot_coreai/schemas/lerobot-coreai.schema.json so it
    is packaged inside the wheel (not left outside the package).
    """
    schema_text = files("lerobot_coreai.schemas").joinpath(
        "lerobot-coreai.schema.json"
    ).read_text()
    return json.loads(schema_text)


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
    coreai_model_id: str | None
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
    allowed_modes: list[str]
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
            coreai_model_id=coreai.get("model_id"),
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
            allowed_modes=safety.get("allowed_modes", ["dry_run", "shadow", "sim", "real"]),
            real_actuation_requires_confirmation=safety.get("real_actuation_requires_confirmation", True),
            raw=data,
        )

    @property
    def parity_passed(self) -> bool:
        """True if action parity was verified and passed.

        Checks not just status but also the metric type and the proves flags, so a
        manifest with status=passed but proves_numeric_fidelity=false (or a non-action
        metric) does not claim action parity.
        """
        return (
            self.evaluation_status == "passed"
            and self.evaluation_metric == "action_parity"
            and self.proves_numeric_fidelity is True
            and self.proves_task_success is False
            and self.proves_robot_safety is False
        )

    @property
    def contracts(self) -> dict[str, Any]:
        """The optional machine-readable contracts block (v1.2.8+), or {}.

        Holds ``action`` / ``batch`` / ``processor`` sub-contracts. Absent in v0
        manifests; parsers fall back to inference when a sub-contract is missing.
        """
        c = self.raw.get("contracts")
        return c if isinstance(c, dict) else {}

    @property
    def lerobot_version_supported(self) -> str:
        """The LeRobot version this artifact was exported against."""
        return self.framework_version

    @property
    def model_id(self) -> str:
        """The runner model id for this artifact.

        Uses the explicit ``coreai.model_id`` from the manifest if present.
        Otherwise derives a best-effort id from the repo_id:
            kevinqz/EVO1-SO100-CoreAI → evo1-so100
        """
        if self.coreai_model_id:
            return self.coreai_model_id
        return derive_model_id(self.policy_repo_id)


def derive_model_id(repo_id: str) -> str:
    """Derive a runner model_id from an HF repo_id.

    Example: kevinqz/EVO1-SO100-CoreAI → evo1-so100
    """
    return (
        repo_id
        .split("/")[-1]
        .replace("-CoreAI", "")
        .lower()
    )


def _looks_local(source: str) -> bool:
    """True if a source string is a filesystem path, not an HF repo id.

    HF repo ids look like ``org/name`` (no leading ./ ~ / and not absolute). A
    local path is absolute or starts with ./ ../ ~ — those must NEVER silently
    fall back to the Hugging Face Hub.
    """
    import os
    return (source.startswith(("/", "./", "../", "~"))
            or os.path.isabs(source))


def resolve_manifest(source: str, *, revision: str = "main"):
    """Resolve a manifest from a local dir/file or an HF repo (discriminated).

    Returns ``(manifest, source_kind, sha256, network_accessed)``. Local sources
    never touch the network; a nonexistent local-looking path fails rather than
    falling back to the Hub.
    """
    from pathlib import Path

    from .errors import DownloadError

    p = Path(source).expanduser()
    # 1. Existing local directory or file.
    if p.is_dir():
        mpath = p / MANIFEST_FILENAME
        if not mpath.is_file():
            raise ManifestError(
                f"{MANIFEST_FILENAME} not found in local directory {p}.",
                repo_id=source)
        data = json.loads(mpath.read_bytes())
        return (LeRobotCoreAIManifest.from_dict(data), "local_directory",
                _sha256_bytes(mpath.read_bytes()), False)
    if p.is_file():
        raw = p.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ManifestError(f"{p} is not valid JSON: {e}") from e
        return (LeRobotCoreAIManifest.from_dict(data), "local_file",
                _sha256_bytes(raw), False)
    # 2. Local-looking but nonexistent → fail closed (no silent HF fallback).
    if _looks_local(source):
        raise ManifestError(
            f"local manifest path does not exist: {source} (refusing to fall "
            "back to the Hugging Face Hub for a local-looking path).",
            repo_id=source)

    # 3. HF repo id.
    url = HF_RAW_BASE.format(repo=source, filename=MANIFEST_FILENAME)
    url = url.replace("/resolve/main/", f"/resolve/{revision}/")
    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
    except httpx.RequestError as e:
        raise DownloadError(f"Failed to fetch {MANIFEST_FILENAME} from {source}: {e}") from e
    if resp.status_code == 404:
        raise ManifestError(
            f"{MANIFEST_FILENAME} not found in {source}. "
            f"This repo may not be a CoreAI-backed LeRobot artifact.", repo_id=source)
    if resp.status_code != 200:
        raise DownloadError(
            f"HTTP {resp.status_code} fetching {MANIFEST_FILENAME} from {source}")
    try:
        data = resp.json()
    except json.JSONDecodeError as e:
        raise ManifestError(f"{MANIFEST_FILENAME} in {source} is not valid JSON: {e}") from e
    return (LeRobotCoreAIManifest.from_dict(data), "hf_repo",
            _sha256_bytes(resp.content), True)


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def load_manifest(repo_id: str, *, revision: str = "main") -> LeRobotCoreAIManifest:
    """Load + validate lerobot-coreai.json from a local dir/file or HF repo.

    Local paths (absolute, ./, ../, ~, or an existing path) are read from disk
    with no network access; a local-looking path that doesn't exist fails rather
    than falling back to the Hub. Everything else is treated as an HF repo id.
    """
    manifest, _kind, _sha, _net = resolve_manifest(repo_id, revision=revision)
    return manifest


def _load_manifest_from_hf(repo_id: str, *, revision: str = "main") -> LeRobotCoreAIManifest:
    from .errors import DownloadError

    url = HF_RAW_BASE.format(repo=repo_id, filename=MANIFEST_FILENAME)
    # Override revision in the URL
    url = url.replace("/resolve/main/", f"/resolve/{revision}/")

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
    except httpx.RequestError as e:
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
