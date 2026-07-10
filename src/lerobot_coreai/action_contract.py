# action_contract.py — explicit action + batch contracts (v1.2.5).
#
# The v1.2.4 compatibility contract honestly reported that select_action's
# semantics differ from LeRobot's (chunk passthrough, not per-timestep). This
# module makes the contract explicit and machine-readable so the runtime can do
# the right thing: chunked policies own a queue; select-next pops one action per
# step. It never claims LeRobot per-timestep semantics unless the contract says
# so. No hardware, no egress.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ACTION_CONTRACT_SCHEMA_VERSION = "lerobot-coreai.action_contract.v1"

_SINGLE = "single"
_CHUNK = "chunk"


@dataclass
class ActionContract:
    representation: str = _CHUNK           # "single" | "chunk"
    horizon: int = 1                       # chunk length (1 for single)
    action_dim: int | None = None
    select_action_semantics: str = "next_action"
    predict_action_chunk_semantics: str = "full_chunk"
    queue_owner: str = "python_bridge"
    reset_clears_queue: bool = True
    temporal_ensembling: bool = False

    def is_chunked(self) -> bool:
        return self.representation == _CHUNK and self.horizon > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "select_action_semantics": self.select_action_semantics,
            "predict_action_chunk_semantics": self.predict_action_chunk_semantics,
            "queue_owner": self.queue_owner,
            "reset_clears_queue": self.reset_clears_queue,
            "temporal_ensembling": self.temporal_ensembling,
        }


@dataclass
class BatchContract:
    supports_batch: bool = False
    max_batch_size: int = 1
    fallback: str = "split_and_stack"      # "split_and_stack" | "reject"

    def to_dict(self) -> dict[str, Any]:
        return {
            "supports_batch": self.supports_batch,
            "max_batch_size": self.max_batch_size,
            "fallback": self.fallback,
        }


def _first_action_shape(manifest) -> list[int] | None:
    """Return the shape of the first action feature, from a dict or manifest obj."""
    # Object form.
    feats = getattr(manifest, "action_features", None)
    if isinstance(feats, dict) and feats:
        first = next(iter(feats.values()))
        shape = getattr(first, "shape", None)
        if shape is None and isinstance(first, dict):
            shape = first.get("shape")
        return list(shape) if shape is not None else None
    # Raw dict form (manifest json).
    if isinstance(manifest, dict):
        af = (manifest.get("action_features")
              or manifest.get("policy", {}).get("action_features"))
        if isinstance(af, dict) and af:
            first = next(iter(af.values()))
            shape = first.get("shape") if isinstance(first, dict) else None
            return list(shape) if shape is not None else None
    return None


def parse_action_contract_from_manifest(manifest) -> ActionContract:
    """Derive an ActionContract, honoring an explicit block or inferring safely.

    An explicit ``action_contract`` in the manifest wins. Otherwise we infer a
    backward-compatible contract: a 2D action shape ``[H, A]`` implies a chunk of
    horizon H; a 1D ``[A]`` implies a single action. We never assert LeRobot
    per-timestep semantics from inference alone — inference only records shape.
    """
    explicit = None
    if isinstance(manifest, dict):
        explicit = manifest.get("action_contract")
    else:
        explicit = getattr(manifest, "action_contract", None)
    if isinstance(explicit, dict):
        return ActionContract(
            representation=explicit.get("representation", _CHUNK),
            horizon=int(explicit.get("horizon", 1)),
            action_dim=explicit.get("action_dim"),
            select_action_semantics=explicit.get("select_action_semantics", "next_action"),
            predict_action_chunk_semantics=explicit.get(
                "predict_action_chunk_semantics", "full_chunk"),
            queue_owner=explicit.get("queue_owner", "python_bridge"),
            reset_clears_queue=explicit.get("reset_clears_queue", True),
            temporal_ensembling=explicit.get("temporal_ensembling", False))

    shape = _first_action_shape(manifest)
    if shape and len(shape) >= 2:
        return ActionContract(representation=_CHUNK, horizon=int(shape[0]),
                              action_dim=int(shape[-1]))
    if shape and len(shape) == 1:
        return ActionContract(representation=_SINGLE, horizon=1,
                              action_dim=int(shape[0]))
    # Unknown shape: default to a single-action contract (safest, non-chunked).
    return ActionContract(representation=_SINGLE, horizon=1, action_dim=None)


def parse_batch_contract_from_manifest(manifest) -> BatchContract:
    explicit = None
    if isinstance(manifest, dict):
        explicit = manifest.get("batch_contract")
    else:
        explicit = getattr(manifest, "batch_contract", None)
    if isinstance(explicit, dict):
        return BatchContract(
            supports_batch=explicit.get("supports_batch", False),
            max_batch_size=int(explicit.get("max_batch_size", 1)),
            fallback=explicit.get("fallback", "split_and_stack"))
    return BatchContract()


def build_action_contract_report(action: ActionContract,
                                 batch: BatchContract) -> dict[str, Any]:
    return {
        "schema_version": ACTION_CONTRACT_SCHEMA_VERSION,
        "action_contract": action.to_dict(),
        "batch_contract": batch.to_dict(),
        "claims": {
            # True only for the project-local bridge's select_next_action, which
            # DOES return a per-timestep action from the queue. The official
            # plugin (v1.3.x) is what will satisfy LeRobot's Tensor(B,A) contract.
            "matches_lerobot_select_action_semantics": (
                action.select_action_semantics == "next_action"),
            "supports_training": False,
            "proves_physical_safety": False,
        },
    }
