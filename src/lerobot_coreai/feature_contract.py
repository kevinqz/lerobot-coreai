# feature_contract.py — FeatureContract v1: stage-bound feature semantics (v1.3.24).
#
# "Every tensor must declare what it means, where it exists in the pipeline, and who
# owns its transformation." v1.3.23 proved shapes/transport for particular fixtures;
# FeatureContract v1 turns that into a declarative, stage-bound, versioned contract:
# dtype, symbolic shape, axes/layout, component names+order, value domain, units,
# coordinate frame, requiredness, batching, temporal semantics and normalization
# OWNERSHIP — per (role, key, stage). Pure Python + JSON; lerobot-free.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rollout_evidence_schema import canonical_json_sha256
from .stages import ACTION_STAGES, OBSERVATION_STAGES

FEATURE_CONTRACT_SCHEMA_VERSION = "lerobot-coreai.feature-contract.v1"

ROLES = ("observation", "action", "context")
MODALITIES = ("vector", "image", "video", "depth", "text", "scalar")
# declared shape symbols (v1: no implicit symbols allowed).
SHAPE_SYMBOLS = ("B", "T", "H", "A", "C", "IH", "IW", "S")
NORMALIZATION_STATES = ("none", "raw", "normalized", "unknown")
NORMALIZATION_OWNERS = (
    "environment", "preprocess_observation", "env_preprocessor",
    "policy_preprocessor", "coreai_transport", "coreai_model",
    "policy_postprocessor", "env_postprocessor",
)
TEMPORAL_KINDS = ("instantaneous", "chunk", "history", "future")


class FeatureContractError(ValueError):
    """Raised when a FeatureContract is malformed or a payload violates it."""


@dataclass(frozen=True)
class ValueDomain:
    finite: bool = True
    minimum: float | None = None
    maximum: float | None = None
    closed_interval: bool = False

    def to_dict(self) -> dict:
        return {"finite": self.finite, "minimum": self.minimum,
                "maximum": self.maximum, "closed_interval": self.closed_interval}


@dataclass(frozen=True)
class NormalizationContract:
    state: str = "unknown"                    # none|raw|normalized|unknown
    method: str | None = None
    owner: str | None = None                  # single owner per boundary
    stats_ref: str | None = None

    def to_dict(self) -> dict:
        return {"state": self.state, "method": self.method,
                "owner": self.owner, "stats_ref": self.stats_ref}


@dataclass(frozen=True)
class FeatureSpec:
    feature_id: str
    key: str
    role: str
    modality: str
    stage: str
    required: bool
    dtype: str
    shape: tuple[Any, ...]                    # ints or declared symbols
    axes: tuple[str, ...]
    layout: str | None
    value_domain: ValueDomain
    units: Any = None                         # str | tuple | None
    coordinate_frame: str | None = None
    names: tuple[str, ...] | None = None      # component names (order-significant)
    normalization: NormalizationContract = field(default_factory=NormalizationContract)

    def to_dict(self) -> dict:
        return {
            "feature_id": self.feature_id, "key": self.key, "role": self.role,
            "modality": self.modality, "stage": self.stage, "required": self.required,
            "dtype": self.dtype, "shape": list(self.shape), "axes": list(self.axes),
            "layout": self.layout, "value_domain": self.value_domain.to_dict(),
            "units": list(self.units) if isinstance(self.units, tuple) else self.units,
            "coordinate_frame": self.coordinate_frame,
            "names": list(self.names) if self.names is not None else None,
            "normalization": self.normalization.to_dict(),
        }


@dataclass(frozen=True)
class FeatureContract:
    contract_id: str
    robot_type: str | None
    policy_path: str | None
    observations: tuple[FeatureSpec, ...]
    actions: tuple[FeatureSpec, ...]
    context: tuple[FeatureSpec, ...] = ()
    processor_stage_contract_sha256: str | None = None
    runtime_support_profile_sha256: str | None = None
    schema_version: str = FEATURE_CONTRACT_SCHEMA_VERSION

    def all_specs(self) -> tuple[FeatureSpec, ...]:
        return self.observations + self.actions + self.context

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "contract_id": self.contract_id,
            "robot_type": self.robot_type, "policy_path": self.policy_path,
            "processor_stage_contract_sha256": self.processor_stage_contract_sha256,
            "runtime_support_profile_sha256": self.runtime_support_profile_sha256,
            "observations": [s.to_dict() for s in self.observations],
            "actions": [s.to_dict() for s in self.actions],
            "context": [s.to_dict() for s in self.context],
            "claims": {"feature_contract_verified": False,
                       "proves_task_success": False, "proves_physical_safety": False},
        }

    def sha256(self) -> str:
        d = self.to_dict()
        d.pop("claims", None)                 # hash the semantics, not the claim flags
        return canonical_json_sha256(d)


def make_feature_id(role: str, key: str, stage: str) -> str:
    return f"{role}:{key}@{stage}"


def _spec_from_dict(d: dict) -> FeatureSpec:
    vd = d.get("value_domain", {}) or {}
    nz = d.get("normalization", {}) or {}
    units = d.get("units")
    return FeatureSpec(
        feature_id=d["feature_id"], key=d["key"], role=d["role"],
        modality=d["modality"], stage=d["stage"], required=bool(d["required"]),
        dtype=d["dtype"], shape=tuple(d["shape"]), axes=tuple(d["axes"]),
        layout=d.get("layout"),
        value_domain=ValueDomain(vd.get("finite", True), vd.get("minimum"),
                                 vd.get("maximum"), vd.get("closed_interval", False)),
        units=tuple(units) if isinstance(units, list) else units,
        coordinate_frame=d.get("coordinate_frame"),
        names=tuple(d["names"]) if d.get("names") is not None else None,
        normalization=NormalizationContract(nz.get("state", "unknown"),
                                            nz.get("method"), nz.get("owner"),
                                            nz.get("stats_ref")))


def feature_contract_from_manifest(
    manifest_features: dict, *, contract_id: str, robot_type: str | None,
    runtime_backend: str = "coreai",
    processor_stage_contract_sha256: str | None = None,
    runtime_support_profile_sha256: str | None = None,
) -> "FeatureContract":
    """Derive a FeatureContract at the provider-input/output stages from a CoreAI
    manifest ``features`` block (v1.3.24a) — so the artifact can bind a contract root
    without hand-authored fixtures. Backend-neutral: the stage is the concrete backend
    runner boundary, the backend is recorded on the contract id."""
    from .stages import ActionStage, ObservationStage
    obs_stage = ObservationStage.COREAI_RUNNER_INPUT.value if runtime_backend == "coreai" \
        else f"{runtime_backend}_runner_input.v1"
    act_stage = ActionStage.COREAI_RUNNER_OUTPUT.value if runtime_backend == "coreai" \
        else f"{runtime_backend}_runner_output.v1"
    observations: list[FeatureSpec] = []
    context: list[FeatureSpec] = []
    for key, spec in (manifest_features.get("observation", {}) or {}).items():
        dtype = spec.get("dtype", "float32")
        required = bool(spec.get("required", True))
        if key == "task" or dtype == "string":
            context.append(FeatureSpec(
                feature_id=make_feature_id("context", key, obs_stage), key=key,
                role="context", modality="text", stage=obs_stage, required=required,
                dtype="string", shape=(), axes=(), layout=None,
                value_domain=ValueDomain(finite=True)))
            continue
        shape = tuple(spec.get("shape", []) or [])
        modality = "image" if dtype in ("image", "video") else "vector"
        names = tuple(spec["names"]) if spec.get("names") else None
        observations.append(FeatureSpec(
            feature_id=make_feature_id("observation", key, obs_stage), key=key,
            role="observation", modality=modality, stage=obs_stage, required=required,
            dtype=("float32" if modality == "image" else dtype), shape=shape,
            axes=tuple(f"d{i}" for i in range(len(shape))), layout=None,
            value_domain=ValueDomain(finite=True), names=names,
            normalization=NormalizationContract("unknown", None, None)))
    actions: list[FeatureSpec] = []
    for key, spec in (manifest_features.get("action", {}) or {}).items():
        shape = tuple(spec.get("shape", []) or [])
        actions.append(FeatureSpec(
            feature_id=make_feature_id("action", key, act_stage), key=key,
            role="action", modality="vector", stage=act_stage, required=True,
            dtype=spec.get("dtype", "float32"), shape=shape,
            axes=tuple(f"d{i}" for i in range(len(shape))), layout=None,
            value_domain=ValueDomain(finite=True),
            names=tuple(spec["names"]) if spec.get("names") else None,
            normalization=NormalizationContract("unknown", None, None)))
    return FeatureContract(
        contract_id=contract_id, robot_type=robot_type, policy_path=None,
        observations=tuple(observations), actions=tuple(actions),
        context=tuple(context),
        processor_stage_contract_sha256=processor_stage_contract_sha256,
        runtime_support_profile_sha256=runtime_support_profile_sha256)


def feature_contract_from_dict(d: dict) -> FeatureContract:
    return FeatureContract(
        contract_id=d["contract_id"], robot_type=d.get("robot_type"),
        policy_path=d.get("policy_path"),
        processor_stage_contract_sha256=d.get("processor_stage_contract_sha256"),
        runtime_support_profile_sha256=d.get("runtime_support_profile_sha256"),
        observations=tuple(_spec_from_dict(s) for s in d.get("observations", [])),
        actions=tuple(_spec_from_dict(s) for s in d.get("actions", [])),
        context=tuple(_spec_from_dict(s) for s in d.get("context", [])),
        schema_version=d.get("schema_version", FEATURE_CONTRACT_SCHEMA_VERSION))
