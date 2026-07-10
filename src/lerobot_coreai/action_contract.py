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


BATCH_CONTRACT_SCHEMA_VERSION = "coreai-batch-contract.v2"
_VALID_FALLBACKS = ("split_and_stack", "reject")
_VALID_CLIENT_MODES = ("native_batch", "split_and_stack")
_VALID_QUEUE_LAYOUTS = ("time_major_batched",)


@dataclass
class BatchContract:
    supports_batch: bool = False
    max_batch_size: int = 1
    fallback: str = "split_and_stack"      # "split_and_stack" | "reject"
    # v2 (v1.3.9): explicit client modes + atomic-commit contract.
    schema_version: str = BATCH_CONTRACT_SCHEMA_VERSION
    supported_client_modes: tuple[str, ...] = ("native_batch", "split_and_stack")
    queue_layout: str = "time_major_batched"
    requires_atomic_commit: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_supports_batch": self.supports_batch,
            "supported_client_modes": list(self.supported_client_modes),
            "max_batch_size": self.max_batch_size,
            "fallback": self.fallback,
            "queue_layout": self.queue_layout,
            "requires_atomic_commit": self.requires_atomic_commit,
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


def _contracts_block(manifest) -> dict[str, Any]:
    """Return the v1 ``contracts`` block from a dict or manifest object, or {}."""
    if isinstance(manifest, dict):
        c = manifest.get("contracts")
    else:
        c = getattr(manifest, "contracts", None)
    return c if isinstance(c, dict) else {}


def parse_action_contract_from_manifest(manifest) -> ActionContract:
    """Derive an ActionContract, honoring an explicit block or inferring safely.

    Precedence: v1 ``contracts.action`` → v0 top-level ``action_contract`` →
    shape inference (2D ``[H, A]`` → chunk of horizon H; 1D ``[A]`` → single).
    Inference never asserts LeRobot per-timestep semantics — it records shape only.
    """
    explicit = _contracts_block(manifest).get("action")
    if not isinstance(explicit, dict):
        if isinstance(manifest, dict):
            explicit = manifest.get("action_contract")
        else:
            explicit = getattr(manifest, "action_contract", None)
    if isinstance(explicit, dict):
        representation = explicit.get("representation", _CHUNK)
        horizon = int(explicit.get("horizon", 1))
        # Fail closed on semantically invalid contracts (v1.3.5): an unknown
        # representation, or single-with-horizon!=1, is a manifest error — not a
        # value to silently coerce.
        if representation not in (_SINGLE, _CHUNK):
            raise ValueError(
                f"invalid action representation {representation!r} "
                f"(expected {_SINGLE!r} or {_CHUNK!r}).")
        if representation == _SINGLE and horizon != 1:
            raise ValueError(
                f"representation=single requires horizon=1, got horizon={horizon}.")
        if representation == _CHUNK and horizon < 1:
            raise ValueError(
                f"representation=chunk requires horizon>=1, got horizon={horizon}.")
        return ActionContract(
            representation=representation,
            horizon=horizon,
            action_dim=explicit.get("action_dim"),
            # v1 uses "selection_semantics"; v0 used "select_action_semantics".
            select_action_semantics=explicit.get(
                "selection_semantics",
                explicit.get("select_action_semantics", "next_action")),
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
    explicit = _contracts_block(manifest).get("batch")
    if not isinstance(explicit, dict):
        if isinstance(manifest, dict):
            explicit = manifest.get("batch_contract")
        else:
            explicit = getattr(manifest, "batch_contract", None)
    if isinstance(explicit, dict):
        # v2 uses "policy_supports_batch"; v1 "runner_supports_batch"; v0 "supports_batch".
        supports = explicit.get(
            "policy_supports_batch",
            explicit.get("runner_supports_batch",
                         explicit.get("supports_batch", False)))
        try:
            max_bs = int(explicit.get("max_batch_size", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"batch contract max_batch_size is not an integer: {exc}") from exc
        if max_bs < 1:
            raise ValueError(
                f"batch contract max_batch_size must be >= 1, got {max_bs}.")
        fallback = explicit.get("fallback", "split_and_stack")
        if fallback not in _VALID_FALLBACKS:
            raise ValueError(
                f"batch contract fallback {fallback!r} not in {_VALID_FALLBACKS}.")
        modes = tuple(explicit.get("supported_client_modes",
                                   ("native_batch", "split_and_stack")))
        for m in modes:
            if m not in _VALID_CLIENT_MODES:
                raise ValueError(
                    f"batch contract client mode {m!r} not in {_VALID_CLIENT_MODES}.")
        layout = explicit.get("queue_layout", "time_major_batched")
        if layout not in _VALID_QUEUE_LAYOUTS:
            raise ValueError(
                f"batch contract queue_layout {layout!r} not in {_VALID_QUEUE_LAYOUTS}.")
        return BatchContract(
            supports_batch=bool(supports), max_batch_size=max_bs, fallback=fallback,
            schema_version=explicit.get("schema_version", BATCH_CONTRACT_SCHEMA_VERSION),
            supported_client_modes=modes, queue_layout=layout,
            requires_atomic_commit=bool(explicit.get("requires_atomic_commit", True)))
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
