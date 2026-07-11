# test_negotiation_algorithm.py — v1.3.20 pure, offline negotiation re-run (P1.2).

import pytest

from lerobot_coreai.negotiation_algorithm import (
    NegotiationError, expected_encoding, expected_negotiation, expected_protocol,
    parse_protocol,
)


def test_parse_protocol():
    assert parse_protocol("coreai-runner.v2") == ("coreai-runner", 2)
    assert parse_protocol("bogus") is None
    assert parse_protocol(None) is None


def test_protocol_equal_to_minimum_ok():
    assert expected_protocol("coreai-runner.v2", "coreai-runner.v2", []) == "coreai-runner.v2"


def test_protocol_below_minimum_fails():
    # the P1.2 example: negotiated v1 while the minimum is v2.
    with pytest.raises(NegotiationError):
        expected_protocol("coreai-runner.v2", "coreai-runner.v1", [])


def test_newer_major_without_backward_compat_fails():
    with pytest.raises(NegotiationError):
        expected_protocol("coreai-runner.v2", "coreai-runner.v3", [])


def test_newer_major_with_declared_backward_compat_ok():
    assert expected_protocol(
        "coreai-runner.v2", "coreai-runner.v3", ["coreai-runner.v2"]) == "coreai-runner.v3"


def test_family_mismatch_fails():
    with pytest.raises(NegotiationError):
        expected_protocol("coreai-runner.v2", "evil-protocol.v2", [])


def test_encoding_not_announced_fails():
    with pytest.raises(NegotiationError):
        expected_encoding("nested_json_v1", ["typed_array_envelope_v1"])


def test_encoding_auto_picks_first_supported_announced():
    assert expected_encoding("auto", ["typed_array_envelope_v1"]) == "typed_array_envelope_v1"
    assert expected_encoding(None, ["nested_json_v1", "typed_array_envelope_v1"]) == "nested_json_v1"


def test_expected_negotiation_roundtrip():
    rec = {"minimum_protocol": "coreai-runner.v2", "runner_protocol": "coreai-runner.v3",
           "runner_backward_compatible_with": ["coreai-runner.v2"],
           "requested_encoding": None, "runner_encodings": ["nested_json_v1"]}
    assert expected_negotiation(rec) == ("coreai-runner.v3", "nested_json_v1")


def test_expected_negotiation_rejects_invalid_record():
    # a self-consistent record that is semantically invalid (below minimum).
    rec = {"minimum_protocol": "coreai-runner.v2", "runner_protocol": "coreai-runner.v1",
           "runner_backward_compatible_with": [], "requested_encoding": None,
           "runner_encodings": ["nested_json_v1"]}
    with pytest.raises(NegotiationError):
        expected_negotiation(rec)
