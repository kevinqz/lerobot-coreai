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


def test_normalization_is_fail_closed_no_string_to_bool():
    # the audit's bug: bool("false") is True. A JSON string must NOT become a bool.
    with pytest.raises(CapabilitiesNormalizationError):
        normalize_capabilities({"supports": {"action": "false"}})


def test_normalization_is_fail_closed_no_string_to_int():
    with pytest.raises(CapabilitiesNormalizationError):
        normalize_capabilities({"action_batching": {"max_batch_size": "4"}})


def test_normalization_rejects_unknown_enum():
    with pytest.raises(CapabilitiesNormalizationError):
        normalize_capabilities({"action_batching": {"slot_isolation": "telepathic"}})


def test_conditional_batching_supported_requires_semantics_and_isolation():
    with pytest.raises(CapabilitiesNormalizationError):   # supported, no semantics
        normalize_capabilities({"action_batching": {"supported": True,
                                                    "max_batch_size": 4,
                                                    "slot_isolation": "independent"}})
    with pytest.raises(CapabilitiesNormalizationError):   # supported, max < 2
        normalize_capabilities({"action_batching": {"supported": True,
                                                    "max_batch_size": 1,
                                                    "semantics": "native",
                                                    "slot_isolation": "independent"}})


def test_conditional_session_scope_requires_session_ids():
    with pytest.raises(CapabilitiesNormalizationError):
        normalize_capabilities({"inference_state": {"scope": "session_scoped",
                                                    "supports_session_ids": False}})


def test_slot_isolation_alias_resolved_and_conflict_fails():
    # only the alias present -> resolved.
    n = normalize_capabilities({"action_batching": {"state_isolation": "independent"}})
    assert n["action_batching"]["slot_isolation"] == "independent"
    # both present with different values -> conflict fails.
    with pytest.raises(CapabilitiesNormalizationError):
        normalize_capabilities({"action_batching": {"slot_isolation": "independent",
                                                    "state_isolation": "shared"}})


def test_normalized_capabilities_is_batch_decision_authority():
    # the typed object duck-types as the capabilities the batch decision reads.
    from lerobot_coreai.capabilities_normalize import typed_normalized_capabilities
    raw = {"supports": {"action": True}, "observation_encodings": ["nested_json_v1"],
           "action_batching": {"supported": True, "max_batch_size": 4,
                               "semantics": "native", "slot_isolation": "independent"},
           "inference_state": {"scope": "stateless", "supports_session_ids": False,
                               "reset_scope": "none"}}
    typed = typed_normalized_capabilities(raw)
    assert typed.supports_batch is True and typed.max_batch_size == 4
    assert typed.action_batching_semantics == "native"
    assert typed.action_batching_slot_isolation == "independent"
    assert typed.action_batching_state_isolation == "independent"   # alias accessor
    assert typed.inference_state_scope == "stateless"


def test_typed_normalized_capabilities_roundtrip():
    from lerobot_coreai.capabilities_normalize import typed_normalized_capabilities
    raw = {"protocol_version": "coreai-runner.v2", "supports": {"action": True},
           "observation_encodings": ["nested_json_v1"],
           "action_batching": {"supported": True, "max_batch_size": 4,
                               "semantics": "native", "slot_isolation": "independent"}}
    typed = typed_normalized_capabilities(raw)
    assert typed.supports_action is True and typed.max_batch_size == 4
    assert typed.slot_isolation == "independent"
    assert normalize_capabilities(raw) == typed.to_dict()   # typed <-> dict parity


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
