# stages.py — canonical observation/action stage vocabulary (v1.3.20).
#
# A FeatureContract (v1.3.21) is only meaningful if it names the EXACT pipeline
# stage a feature lives in — "float32 CHW [0,1]" means nothing without knowing
# whether it is the environment-raw observation, the policy-preprocessor output, or
# the CoreAI-runner input. This base vocabulary + a ProcessorStageContract give
# those tensors a single, typed language. Pure Python, lerobot-free.
#
# NOTE (v1.3.20): this vocabulary is introduced as an ADDITIVE foundation. The full
# migration off the legacy ``raw_lerobot_observation`` ownership string across the
# manifest + all consumers is a wide, isolated rename deferred to v1.3.21.

from __future__ import annotations

from enum import Enum


class ObservationStage(str, Enum):
    """Where an observation tensor sits in the LeRobot -> CoreAI pipeline."""
    ENVIRONMENT_RAW = "environment_raw.v1"
    LEROBOT_PREPROCESS_OUTPUT = "lerobot_preprocess_observation_output.v1"
    ENV_PROCESSOR_OUTPUT = "lerobot_env_preprocessor_output.v1"
    POLICY_PROCESSOR_OUTPUT = "lerobot_policy_preprocessor_output.v1"
    COREAI_RUNNER_INPUT = "coreai_runner_input.v1"


class ActionStage(str, Enum):
    """Where an action tensor sits in the CoreAI -> LeRobot pipeline."""
    COREAI_RUNNER_OUTPUT = "coreai_runner_output.v1"
    ASSEMBLED_ACTION_CHUNK = "coreai_assembled_action_chunk.v1"
    VALIDATED_ACTION_CHUNK = "coreai_validated_action_chunk.v1"
    QUEUE_COMMITTED_CHUNK = "coreai_queue_committed_chunk.v1"
    SELECTED_POLICY_ACTION = "lerobot_selected_policy_action.v1"
    POLICY_POSTPROCESSOR_OUTPUT = "lerobot_policy_postprocessor_output.v1"
    ENVIRONMENT_ACTION = "environment_action.v1"


# v1.3.24a backend-neutral layer: the certification contracts must not be locked to
# CoreAI, so that MLX / PyTorch-reference providers can plug in later WITHOUT a
# destructive migration. A canonical provider stage names the pipeline position; the
# backend stage names the concrete runtime realization of it. The legacy coreai_*
# stages stay valid (readable) — they ARE the coreai backend_stage values.
RUNTIME_BACKENDS = ("coreai", "mlx", "pytorch_reference")


class RuntimeProviderStage(str, Enum):
    """Backend-neutral canonical pipeline positions (any runtime provider)."""
    PROVIDER_INPUT = "runtime_provider_input.v1"
    PROVIDER_OUTPUT = "runtime_provider_output.v1"


# canonical <- concrete backend_stage mapping (coreai today; mlx/pytorch later).
_BACKEND_STAGE_TO_CANONICAL = {
    ObservationStage.COREAI_RUNNER_INPUT.value: RuntimeProviderStage.PROVIDER_INPUT.value,
    ActionStage.COREAI_RUNNER_OUTPUT.value: RuntimeProviderStage.PROVIDER_OUTPUT.value,
    "mlx_module_input.v1": RuntimeProviderStage.PROVIDER_INPUT.value,
    "mlx_module_output.v1": RuntimeProviderStage.PROVIDER_OUTPUT.value,
}

OBSERVATION_STAGES = tuple(s.value for s in ObservationStage)
ACTION_STAGES = tuple(s.value for s in ActionStage)
PROVIDER_STAGES = tuple(s.value for s in RuntimeProviderStage)
STAGE_TRANSFORMS = ("identity", "nontrivial")


def canonical_provider_stage(backend_stage: str) -> str | None:
    """The backend-neutral canonical stage for a concrete backend stage, if any."""
    return _BACKEND_STAGE_TO_CANONICAL.get(backend_stage)


PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION = "coreai-processor-stage-contract.v1"
_STAGE_SIDE_PROPS = {
    "source": {"type": "string"}, "target": {"type": "string"},
    "transform": {"enum": list(STAGE_TRANSFORMS)},
    # v1.3.24a additive backend-neutral annotations (optional, back-compatible).
    "canonical_source": {"enum": list(PROVIDER_STAGES) + [None]},
    "canonical_target": {"enum": list(PROVIDER_STAGES) + [None]},
    "backend_owner": {"type": ["string", "null"]},
}
PROCESSOR_STAGE_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "observation", "action"],
    "properties": {
        "schema_version": {"const": PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION},
        "runtime_backend": {"enum": list(RUNTIME_BACKENDS)},
        "observation": {
            "type": "object", "additionalProperties": False,
            "required": ["source", "target", "transform"],
            "properties": {**_STAGE_SIDE_PROPS,
                           "source": {"enum": list(OBSERVATION_STAGES)},
                           "target": {"enum": list(OBSERVATION_STAGES)}}},
        "action": {
            "type": "object", "additionalProperties": False,
            "required": ["source", "target", "transform"],
            "properties": {**_STAGE_SIDE_PROPS,
                           "source": {"enum": list(ACTION_STAGES)},
                           "target": {"enum": list(ACTION_STAGES)}}},
    },
}


def identity_transition_preserves_hash(source_sha256: str, target_sha256: str,
                                       transform: str) -> bool:
    """For an ``identity`` transform the input and output hashes MUST be equal;
    a ``nontrivial`` transform is allowed to change them (parity is proven later)."""
    if transform == "identity":
        return source_sha256 == target_sha256
    return transform == "nontrivial"


# v1.3.24 Phase 0: map the legacy processor-contract ownership strings
# (`expects`/`returns`) onto canonical stages, so a canonical ProcessorStageContract
# v1 can be produced from a legacy artifact without inventing semantics. The legacy
# strings remain accepted only in the READER; new writers emit the enum forms.
_LEGACY_EXPECTS_TO_SOURCE = {
    "raw_lerobot_observation": ObservationStage.LEROBOT_PREPROCESS_OUTPUT.value,
    "policy_preprocessed_observation": ObservationStage.POLICY_PROCESSOR_OUTPUT.value,
}
_LEGACY_RETURNS_TO_TARGET = {
    "postprocessed_environment_action": ActionStage.ENVIRONMENT_ACTION.value,
    "postprocessed_action": ActionStage.ENVIRONMENT_ACTION.value,
    "normalized_action": ActionStage.VALIDATED_ACTION_CHUNK.value,
}


def build_processor_stage_contract(*, expects: str, returns: str,
                                   transform: str = "identity",
                                   runtime_backend: str = "coreai") -> dict:
    """Canonical ProcessorStageContract v1 derived from a legacy processor contract.

    ``expects`` is what the runner consumes (its observation source stage → the
    provider input); ``returns`` is what the runner emits (provider output → the
    environment/validated action stage). Unknown legacy values fail closed. The
    contract is backend-neutral: it records ``runtime_backend`` + the canonical
    provider stage for each concrete backend boundary (v1.3.24a)."""
    source = _LEGACY_EXPECTS_TO_SOURCE.get(expects)
    target_action = _LEGACY_RETURNS_TO_TARGET.get(returns)
    if source is None or target_action is None:
        raise ValueError(
            f"cannot map legacy processor contract expects={expects!r} "
            f"returns={returns!r} to canonical stages.")
    if runtime_backend not in RUNTIME_BACKENDS:
        raise ValueError(f"unknown runtime_backend {runtime_backend!r}")
    obs_target = ObservationStage.COREAI_RUNNER_INPUT.value
    act_source = ActionStage.COREAI_RUNNER_OUTPUT.value
    return {
        "schema_version": PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION,
        "runtime_backend": runtime_backend,
        "observation": {"source": source, "target": obs_target,
                        "transform": transform,
                        "canonical_target": canonical_provider_stage(obs_target),
                        "backend_owner": f"{runtime_backend}_transport"},
        "action": {"source": act_source, "target": target_action,
                   "transform": transform,
                   "canonical_source": canonical_provider_stage(act_source),
                   "backend_owner": f"{runtime_backend}_model"},
    }


def processor_stage_contract_sha256(contract: dict) -> str:
    from .rollout_evidence_schema import canonical_json_sha256
    return canonical_json_sha256(contract)
