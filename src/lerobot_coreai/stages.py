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


OBSERVATION_STAGES = tuple(s.value for s in ObservationStage)
ACTION_STAGES = tuple(s.value for s in ActionStage)
STAGE_TRANSFORMS = ("identity", "nontrivial")

PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION = "coreai-processor-stage-contract.v1"
PROCESSOR_STAGE_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "observation", "action"],
    "properties": {
        "schema_version": {"const": PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION},
        "observation": {
            "type": "object", "additionalProperties": False,
            "required": ["source", "target", "transform"],
            "properties": {"source": {"enum": list(OBSERVATION_STAGES)},
                           "target": {"enum": list(OBSERVATION_STAGES)},
                           "transform": {"enum": list(STAGE_TRANSFORMS)}}},
        "action": {
            "type": "object", "additionalProperties": False,
            "required": ["source", "target", "transform"],
            "properties": {"source": {"enum": list(ACTION_STAGES)},
                           "target": {"enum": list(ACTION_STAGES)},
                           "transform": {"enum": list(STAGE_TRANSFORMS)}}},
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
    "postprocessed_action": ActionStage.ENVIRONMENT_ACTION.value,
    "normalized_action": ActionStage.VALIDATED_ACTION_CHUNK.value,
}


def build_processor_stage_contract(*, expects: str, returns: str,
                                   transform: str = "identity") -> dict:
    """Canonical ProcessorStageContract v1 derived from a legacy processor contract.

    ``expects`` is what the runner consumes (its observation source stage → the
    CoreAI runner input); ``returns`` is what the runner emits (CoreAI runner output
    → the environment/validated action stage). Unknown legacy values fail closed."""
    source = _LEGACY_EXPECTS_TO_SOURCE.get(expects)
    target_action = _LEGACY_RETURNS_TO_TARGET.get(returns)
    if source is None or target_action is None:
        raise ValueError(
            f"cannot map legacy processor contract expects={expects!r} "
            f"returns={returns!r} to canonical stages.")
    return {
        "schema_version": PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION,
        "observation": {"source": source,
                        "target": ObservationStage.COREAI_RUNNER_INPUT.value,
                        "transform": transform},
        "action": {"source": ActionStage.COREAI_RUNNER_OUTPUT.value,
                   "target": target_action, "transform": transform},
    }


def processor_stage_contract_sha256(contract: dict) -> str:
    from .rollout_evidence_schema import canonical_json_sha256
    return canonical_json_sha256(contract)
