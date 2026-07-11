# test_capabilities_and_support.py — v1.3.21 capabilities normalization +
# RuntimeSupportProfile.

import jsonschema
import pytest

from lerobot_coreai.capabilities_normalize import (
    CapabilitiesNormalizationError, NORMALIZED_CAPABILITIES_SCHEMA,
    normalize_capabilities, normalized_capabilities_sha256,
)
from lerobot_coreai.runtime_support import (
    RUNTIME_SUPPORT_SCHEMA, runtime_support_profile,
)


def test_normalization_adds_defaults_and_sorts():
    raw = {"runtime": "coreai-runner", "protocol_version": "coreai-runner.v2",
           "observation_encodings": ["typed_array_envelope_v1", "nested_json_v1"],
           "supports": {"action": True}}
    norm = normalize_capabilities(raw)
    jsonschema.validate(norm, NORMALIZED_CAPABILITIES_SCHEMA)
    assert norm["observation_encodings"] == ["nested_json_v1", "typed_array_envelope_v1"]
    assert norm["backward_compatible_with"] == []
    assert norm["supports_action"] is True
    assert norm["action_batching"]["supported"] is False


def test_normalization_is_order_independent_hash():
    a = normalized_capabilities_sha256(
        {"observation_encodings": ["a", "b"], "supports": {"action": True}})
    b = normalized_capabilities_sha256(
        {"supports": {"action": True}, "observation_encodings": ["b", "a"]})
    assert a == b


def test_normalization_rejects_duplicates():
    with pytest.raises(CapabilitiesNormalizationError):
        normalize_capabilities({"observation_encodings": ["nested_json_v1",
                                                          "nested_json_v1"]})


def test_runtime_support_profile_schema_valid():
    jsonschema.validate(runtime_support_profile(), RUNTIME_SUPPORT_SCHEMA)


def test_runtime_support_profile_excludes_unsupported_batch_states():
    prof = runtime_support_profile()
    assert prof["batch"]["native_slot_isolation"] == ["independent"]
    assert "global" not in prof["batch"]["split_state_scopes"]
    assert "session_scoped" not in prof["batch"]["split_state_scopes"]
    # a tampered profile that admits native `shared` must fail the schema.
    bad = runtime_support_profile()
    bad["batch"]["native_slot_isolation"] = ["shared"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, RUNTIME_SUPPORT_SCHEMA)


def test_unified_negotiation_rejects_unsupported_policy():
    from lerobot_coreai.negotiation_algorithm import (
        NegotiationError, negotiate_runner_contract,
    )
    with pytest.raises(NegotiationError):
        negotiate_runner_contract(
            selection_policy="highest_compatible", requested_protocol=None,
            minimum_protocol="coreai-runner.v2", runner_protocol="coreai-runner.v2",
            runner_backward_compatible_with=[], requested_encoding=None,
            runner_encodings=["nested_json_v1"])
