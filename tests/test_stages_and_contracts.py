# test_stages_and_contracts.py — v1.3.20 canonical stage vocabulary + single
# BatchContract v3 schema source.

import jsonschema
import pytest

from lerobot_coreai.stages import (
    ActionStage, ObservationStage, PROCESSOR_STAGE_CONTRACT_SCHEMA,
    PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION, identity_transition_preserves_hash,
)


def test_stage_enums_are_typed_and_stable():
    assert ObservationStage.POLICY_PROCESSOR_OUTPUT.value == \
        "lerobot_policy_preprocessor_output.v1"
    assert ActionStage.ENVIRONMENT_ACTION.value == "environment_action.v1"
    # StrEnum-style: comparable to its string value.
    assert ObservationStage.COREAI_RUNNER_INPUT == "coreai_runner_input.v1"


def test_processor_stage_contract_validates():
    contract = {
        "schema_version": PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION,
        "observation": {"source": ObservationStage.POLICY_PROCESSOR_OUTPUT.value,
                        "target": ObservationStage.COREAI_RUNNER_INPUT.value,
                        "transform": "identity"},
        "action": {"source": ActionStage.COREAI_RUNNER_OUTPUT.value,
                   "target": ActionStage.ENVIRONMENT_ACTION.value,
                   "transform": "identity"}}
    jsonschema.validate(contract, PROCESSOR_STAGE_CONTRACT_SCHEMA)


def test_processor_stage_contract_rejects_unknown_stage():
    bad = {"schema_version": PROCESSOR_STAGE_CONTRACT_SCHEMA_VERSION,
           "observation": {"source": "not_a_stage", "target": "coreai_runner_input.v1",
                           "transform": "identity"},
           "action": {"source": "coreai_runner_output.v1",
                      "target": "environment_action.v1", "transform": "identity"}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, PROCESSOR_STAGE_CONTRACT_SCHEMA)


def test_identity_transition_preserves_hash():
    assert identity_transition_preserves_hash("sha256:a", "sha256:a", "identity") is True
    assert identity_transition_preserves_hash("sha256:a", "sha256:b", "identity") is False
    assert identity_transition_preserves_hash("sha256:a", "sha256:b", "nontrivial") is True


def test_batch_contract_v3_single_schema_source():
    # the canonical serialization validates against the single shared schema file.
    from lerobot_coreai.action_contract import (
        BatchContract, validate_batch_contract_v3,
    )
    bc = BatchContract(schema_version="coreai-batch-contract.v3", native_supported=True,
                       native_max_batch_size=4, split_supported=True,
                       split_max_batch_size=4, authoritative=True)
    validate_batch_contract_v3(bc.to_dict())      # must not raise


def test_batch_contract_v3_schema_rejects_bad_slot_isolation():
    from lerobot_coreai.action_contract import validate_batch_contract_v3
    bad = {"schema_version": "coreai-batch-contract.v3",
           "native_batch": {"supported": True, "max_batch_size": 2,
                            "required_slot_isolation": "telepathic"},
           "client_split": {"supported": False, "max_batch_size": 1,
                            "allowed_state_scopes": ["stateless"]},
           "fallback": "split_and_stack",
           "queue": {"layout": "time_major_batched",
                     "commit_semantics": "atomic_queue_commit"}}
    with pytest.raises(jsonschema.ValidationError):
        validate_batch_contract_v3(bad)
