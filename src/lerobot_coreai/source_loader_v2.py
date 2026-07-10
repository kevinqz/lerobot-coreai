# source_loader_v2.py — load a PyTorch LeRobot policy via the OFFICIAL API (v1.2.6).
#
# The v0.5 loader called make_policy with a string policy_type and fell back to
# instantiating the abstract PreTrainedPolicy base — neither is valid for LeRobot
# 0.6.x. This loads the source policy the way LeRobot itself does:
#   cfg = PreTrainedConfig.from_pretrained(policy_path)
#   ds_meta = LeRobotDatasetMetadata(dataset_repo_id)
#   policy = make_policy(cfg, ds_meta=ds_meta)
#   pre, post = make_pre_post_processors(cfg, pretrained_path=..., dataset_stats=...,
#                                        dataset_meta=...)
# Each stage is a thin, mockable wrapper so the flow is testable without LeRobot,
# and live-exercised in the stable CI job. No hardware, no egress, no training.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import __version__
from .errors import CoreAIPolicyError

SOURCE_POLICY_LOAD_SCHEMA_VERSION = "lerobot-coreai.source_policy_load.v0"


class SourceLoaderError(CoreAIPolicyError):
    """Raised when the official source-policy load fails; carries the stage."""

    def __init__(self, stage: str, message: str):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage


@dataclass
class SourcePolicyBundle:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    dataset_metadata: Any
    config: Any


# --- Thin, mockable wrappers over the official LeRobot API ---

def _load_config(policy_path: str):  # pragma: no cover - needs lerobot
    from lerobot.configs import PreTrainedConfig  # type: ignore
    return PreTrainedConfig.from_pretrained(policy_path)


def _load_dataset_metadata(dataset_repo_id: str):  # pragma: no cover - needs lerobot
    try:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata  # type: ignore
    except Exception:
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # type: ignore
    return LeRobotDatasetMetadata(dataset_repo_id)


def _make_policy(cfg, ds_meta):  # pragma: no cover - needs lerobot
    from lerobot.policies.factory import make_policy  # type: ignore
    # Official signature: make_policy(cfg, ds_meta=..., env_cfg=...). Never a string.
    return make_policy(cfg, ds_meta=ds_meta)


def _make_processors(cfg, policy_path, ds_meta):  # pragma: no cover - needs lerobot
    from lerobot.policies.factory import make_pre_post_processors  # type: ignore
    return make_pre_post_processors(
        cfg, pretrained_path=policy_path,
        dataset_stats=getattr(ds_meta, "stats", None), dataset_meta=ds_meta)


def load_source_policy(policy_path: str, dataset_repo_id: str) -> SourcePolicyBundle:
    """Load a source PyTorch policy + official processors. Fails with the stage."""
    try:
        cfg = _load_config(policy_path)
    except Exception as e:
        raise SourceLoaderError("config", f"PreTrainedConfig.from_pretrained failed: {e}")
    try:
        ds_meta = _load_dataset_metadata(dataset_repo_id)
    except Exception as e:
        raise SourceLoaderError("dataset_meta", f"LeRobotDatasetMetadata failed: {e}")
    try:
        policy = _make_policy(cfg, ds_meta)
    except Exception as e:
        raise SourceLoaderError("policy", f"make_policy failed: {e}")
    try:
        pre, post = _make_processors(cfg, policy_path, ds_meta)
    except Exception as e:
        raise SourceLoaderError("processors", f"make_pre_post_processors failed: {e}")
    return SourcePolicyBundle(policy=policy, preprocessor=pre, postprocessor=post,
                              dataset_metadata=ds_meta, config=cfg)


def build_source_load_report(policy_path: str, dataset_repo_id: str, *,
                             bundle: SourcePolicyBundle | None,
                             error: SourceLoaderError | None = None) -> dict[str, Any]:
    ok = bundle is not None and error is None
    return {
        "schema_version": SOURCE_POLICY_LOAD_SCHEMA_VERSION,
        "lerobot_coreai_version": __version__,
        "ok": ok,
        "policy_path": policy_path,
        "dataset_repo_id": dataset_repo_id,
        "loader": {
            "pretrained_config_loaded": ok or (error is not None and error.stage not in ("config",)),
            "dataset_metadata_loaded": ok or (error is not None and error.stage not in ("config", "dataset_meta")),
            "make_policy_used": ok or (error is not None and error.stage in ("processors",)),
            "preprocessors_loaded": ok,
            "postprocessors_loaded": ok,
            "used_official_api": True,
            "instantiated_pretrained_base": False,
            "used_string_policy_type": False,
        },
        "failed_stage": error.stage if error else None,
        "claims": {
            "proves_source_policy_loaded_with_official_lerobot_api": ok,
            "proves_training_support": False,
            "proves_physical_safety": False,
        },
    }
