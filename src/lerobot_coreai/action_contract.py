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


BATCH_CONTRACT_SCHEMA_VERSION = "coreai-batch-contract.v3"


def load_batch_contract_v3_schema() -> dict:
    """The SINGLE canonical BatchContract v3 JSON Schema (v1.3.20). One file
    (``schemas/batch-contract-v3.schema.json``) is the source of truth for every
    consumer; no consumer keeps a divergent copy."""
    import json
    from importlib.resources import files
    return json.loads(files("lerobot_coreai.schemas").joinpath(
        "batch-contract-v3.schema.json").read_text())


def validate_batch_contract_v3(obj: dict) -> None:
    """Validate a canonical v3 batch-contract dict against the shared schema."""
    import jsonschema
    jsonschema.validate(obj, load_batch_contract_v3_schema())
_VALID_FALLBACKS = ("split_and_stack", "reject")
_VALID_CLIENT_MODES = ("native_batch", "split_and_stack")
_VALID_QUEUE_LAYOUTS = ("time_major_batched",)
_VALID_COMMIT_SEMANTICS = ("atomic_queue_commit",)
_VALID_SLOT_ISOLATION = ("independent", "shared", "unknown")
_VALID_STATE_SCOPES = ("stateless", "request_scoped", "session_scoped", "global")
DEFAULT_OBSERVATION_STAGE = "lerobot_policy_preprocessor_output.v1"


@dataclass
class BatchContract:
    """Authoritative batch contract (v3, v1.3.10): native and split modeled apart.

    v3 separates the native batch capacity (one batched request) from the client
    split capacity (B single requests). Legacy flat v0/v2 blocks are still read for
    B=1 back-compat, but only a v3 contract with both modes explicitly declared can
    authorize B>1 (``authoritative``).
    """
    schema_version: str = BATCH_CONTRACT_SCHEMA_VERSION
    native_supported: bool = False
    native_max_batch_size: int = 1
    native_slot_isolation: str = "independent"   # required slot isolation for native
    split_supported: bool = False
    split_max_batch_size: int = 1
    split_allowed_scopes: tuple[str, ...] = ("stateless", "request_scoped")
    fallback: str = "split_and_stack"            # "split_and_stack" | "reject"
    queue_layout: str = "time_major_batched"
    commit_semantics: str = "atomic_queue_commit"
    observation_stage: str = DEFAULT_OBSERVATION_STAGE
    authoritative: bool = False                   # True only for an explicit v3 block

    # Back-compat convenience (max across modes) for older callers/reports.
    @property
    def supports_batch(self) -> bool:
        return self.native_supported or self.split_supported

    @property
    def max_batch_size(self) -> int:
        return max(self.native_max_batch_size, self.split_max_batch_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "native_batch": {
                "supported": self.native_supported,
                "max_batch_size": self.native_max_batch_size,
                "required_slot_isolation": self.native_slot_isolation,
            },
            "client_split": {
                "supported": self.split_supported,
                "max_batch_size": self.split_max_batch_size,
                "allowed_state_scopes": list(self.split_allowed_scopes),
            },
            "fallback": self.fallback,
            "queue": {"layout": self.queue_layout,
                      "commit_semantics": self.commit_semantics},
            "observation_stage": self.observation_stage,
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


def _int_ge1(value: Any, name: str, default: int = 1) -> int:
    if value is None:
        value = default
    # No str->int coercion (P1.2): a JSON "4" is a contract error, not an int.
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be a JSON integer, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _strict_contract_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(
            f"{name} must be a JSON boolean, got {type(value).__name__} {value!r}.")
    return value


def parse_batch_contract_from_manifest(manifest) -> BatchContract:
    """Parse an authoritative v3 batch contract, or read legacy v0/v2 for B=1.

    A v3 block (``native_batch`` / ``client_split`` present, or
    ``schema_version == coreai-batch-contract.v3``) yields an ``authoritative``
    contract that can gate B>1. Legacy flat blocks are read permissively but stay
    non-authoritative (B>1 refuses to certify on them).
    """
    explicit = _contracts_block(manifest).get("batch")
    if not isinstance(explicit, dict):
        if isinstance(manifest, dict):
            explicit = manifest.get("batch_contract")
        else:
            explicit = getattr(manifest, "batch_contract", None)
    if not isinstance(explicit, dict):
        return BatchContract()

    fallback = explicit.get("fallback", "split_and_stack")
    if fallback not in _VALID_FALLBACKS:
        raise ValueError(f"batch fallback {fallback!r} not in {_VALID_FALLBACKS}.")

    is_v3 = (explicit.get("schema_version") == BATCH_CONTRACT_SCHEMA_VERSION
             or "native_batch" in explicit or "client_split" in explicit)
    if is_v3:
        native = explicit.get("native_batch", {}) or {}
        split = explicit.get("client_split", {}) or {}
        queue = explicit.get("queue", {}) or {}
        layout = queue.get("layout", "time_major_batched")
        commit = queue.get("commit_semantics", "atomic_queue_commit")
        if layout not in _VALID_QUEUE_LAYOUTS:
            raise ValueError(f"queue.layout {layout!r} not in {_VALID_QUEUE_LAYOUTS}.")
        if commit not in _VALID_COMMIT_SEMANTICS:
            raise ValueError(
                f"queue.commit_semantics {commit!r} not in {_VALID_COMMIT_SEMANTICS}.")
        slot_iso = native.get("required_slot_isolation", "independent")
        if slot_iso not in _VALID_SLOT_ISOLATION:
            raise ValueError(
                f"native required_slot_isolation {slot_iso!r} not in {_VALID_SLOT_ISOLATION}.")
        scopes = tuple(split.get("allowed_state_scopes", ("stateless", "request_scoped")))
        for s in scopes:
            if s not in _VALID_STATE_SCOPES:
                raise ValueError(f"split allowed scope {s!r} not in {_VALID_STATE_SCOPES}.")
        native_supported = _strict_contract_bool(
            native.get("supported", False), "native_batch.supported")
        split_supported = _strict_contract_bool(
            split.get("supported", False), "client_split.supported")
        # v1.3.11 (P1.1): the runtime only certifies INDEPENDENT native slots.
        if native_supported and slot_iso != "independent":
            raise ValueError(
                "native_batch.required_slot_isolation must be 'independent' when "
                f"native batch is supported, got {slot_iso!r}.")
        return BatchContract(
            schema_version=explicit.get("schema_version", BATCH_CONTRACT_SCHEMA_VERSION),
            native_supported=native_supported,
            native_max_batch_size=_int_ge1(native.get("max_batch_size"),
                                           "native_batch.max_batch_size"),
            native_slot_isolation=slot_iso,
            split_supported=split_supported,
            split_max_batch_size=_int_ge1(split.get("max_batch_size"),
                                          "client_split.max_batch_size"),
            split_allowed_scopes=scopes,
            fallback=fallback, queue_layout=layout, commit_semantics=commit,
            observation_stage=explicit.get("observation_stage", DEFAULT_OBSERVATION_STAGE),
            authoritative=True)

    # Legacy flat (v0/v2): read permissively, NOT authoritative for B>1.
    supports = bool(explicit.get(
        "policy_supports_batch",
        explicit.get("runner_supports_batch", explicit.get("supports_batch", False))))
    max_bs = _int_ge1(explicit.get("max_batch_size"), "batch max_batch_size")
    modes = tuple(explicit.get("supported_client_modes",
                               ("native_batch", "split_and_stack")))
    for m in modes:
        if m not in _VALID_CLIENT_MODES:
            raise ValueError(f"client mode {m!r} not in {_VALID_CLIENT_MODES}.")
    return BatchContract(
        schema_version=explicit.get("schema_version", "coreai-batch-contract.v2"),
        native_supported=supports and "native_batch" in modes,
        native_max_batch_size=max_bs,
        split_supported=supports and "split_and_stack" in modes,
        split_max_batch_size=max_bs, fallback=fallback, authoritative=False)


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
