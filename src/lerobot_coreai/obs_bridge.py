# obs_bridge.py — observation pipeline bridge check (v1.1.5).
#
# The most common LeRobot<->CoreAI mismatch is not the policy, it is observation
# normalization. This turns the v1.0.4 real-mode observation config into a
# general check: take a sample observation (a LeRobotDataset frame or a robot
# observation) and confirm it becomes exactly the observation dict the CoreAI
# manifest expects — reporting required-key presence, state shape, image-key
# resolution, and task handling. Nothing is dropped silently. Proves the mapping
# for the sample only — not task success or physical safety.

from __future__ import annotations

from typing import Any

from .errors import CoreAIPolicyError
from .observation_adapters import ObservationAdapterConfig, adapt_observation

OBS_BRIDGE_SCHEMA_VERSION = "lerobot-coreai.obs_bridge.v0"


def load_dataset_frame(dataset_repo_id: str, frame_index: int = 0) -> dict[str, Any]:  # pragma: no cover - needs lerobot+net
    """Load one LeRobotDataset frame as a raw observation dict."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
    ds = LeRobotDataset(dataset_repo_id)
    item = ds[frame_index]
    # Keep only observation-shaped keys and 'task'; values pass through.
    return {k: v for k, v in dict(item).items()
            if k.startswith("observation") or k == "task"}


def _image_feature_names(manifest) -> list[str]:
    return [n for n in manifest.observation_features
            if "image" in n or n.startswith("observation.images")]


def _state_feature(manifest, state_key: str):
    return manifest.observation_features.get(state_key)


def evaluate_obs_bridge(
    raw: dict[str, Any],
    config: ObservationAdapterConfig,
    *,
    manifest,
    policy_path: str | None = None,
    input_source: str = "sample",
    frame_index: int | None = None,
) -> dict[str, Any]:
    """Adapt a sample observation and report how it maps onto the manifest.

    Never raises: adaptation failures become failed checks. Sends no action.
    """
    from . import __version__
    checks: list[dict[str, Any]] = []

    def _c(name, passed, detail="", severity="required"):
        checks.append({"name": name, "passed": bool(passed), "severity": severity,
                       "detail": detail})

    adapted = None
    try:
        adapted = adapt_observation(dict(raw), config, manifest=manifest)
        _c("required_keys_present", True)
    except CoreAIPolicyError as e:
        _c("required_keys_present", False, str(e))

    obs = adapted.observation if adapted else {}

    # State shape compatibility.
    state_spec = _state_feature(manifest, config.state_key)
    if state_spec is not None and state_spec.shape is not None:
        val = obs.get(config.state_key)
        if isinstance(val, (list, tuple)):
            want = int(state_spec.shape[-1])
            ok = len(val) == want
            _c("state_shape_compatible", ok,
               "" if ok else f"{config.state_key}: got {len(val)}, want {want}")
        else:
            _c("state_shape_compatible", val is not None,
               "state present but not a vector" if val is not None
               else f"{config.state_key} missing")
    else:
        _c("state_shape_compatible", True, "no state shape constraint",
           severity="info")

    # Image keys resolved.
    image_names = _image_feature_names(manifest)
    if image_names:
        missing = [n for n in image_names if n not in obs]
        _c("image_keys_resolved", not missing,
           "" if not missing else f"missing image keys: {missing}")
    else:
        _c("image_keys_resolved", True, "no image features in manifest",
           severity="info")

    # Task handling — explicit, never silently defaulted.
    task_required = config.require_task or ("task" in manifest.observation_features)
    if task_required:
        _c("task_present", "task" in obs,
           "task supplied by config" if config.task is not None else "")
    else:
        _c("task_present", True, "task not required", severity="info")

    # No silent drop: report kept/dropped explicitly.
    dropped: list[str] = []
    if config.drop_unknown_keys:
        dropped = [k for k in raw if k not in obs]
    _c("no_silent_drop", True,
       f"dropped: {dropped}" if dropped else "nothing dropped", severity="info")

    ok = all(c["passed"] for c in checks if c["severity"] == "required")
    return {
        "schema_version": OBS_BRIDGE_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "policy_path": policy_path,
        "input_source": input_source,
        "frame_index": frame_index,
        "keys_present": adapted.keys_present if adapted else [],
        "keys_missing": adapted.keys_missing if adapted else [],
        "dropped_keys": dropped,
        "warnings": adapted.warnings if adapted else [],
        "checks": checks,
        "claims": {
            "proves_observation_mapping_valid_for_sample": ok,
            "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }
