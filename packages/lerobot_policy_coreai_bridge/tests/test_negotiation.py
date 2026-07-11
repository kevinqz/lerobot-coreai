# test_negotiation.py — real observation-encoding negotiation (v1.3.4).

from dataclasses import dataclass

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_policy_coreai_bridge.negotiation import (
    NegotiatedRunnerProtocol,
    negotiate_observation_encoding,
    negotiate_runner_protocol,
)
from lerobot_policy_coreai_bridge.transport import (
    NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1,
)


@dataclass
class _Caps:
    observation_encodings: tuple = ()
    protocol_version: str | None = None
    supports_batch: bool = False
    max_batch_size: int | None = None
    backward_compatible_with: tuple = ()


def test_runtime_delegates_to_base_negotiation_primitive():
    # v1.3.22 (P1.1): the runtime adapter and the base primitive must agree — the
    # runtime carries no parallel selection logic.
    from lerobot_coreai.negotiation_algorithm import negotiate_runner_contract
    caps = _Caps(observation_encodings=(NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1),
                 protocol_version="coreai-runner.v3",
                 backward_compatible_with=("coreai-runner.v2",))
    runtime = negotiate_runner_protocol(
        requested_encoding="auto", capabilities=caps,
        minimum_protocol="coreai-runner.v2")
    base = negotiate_runner_contract(
        selection_policy="minimum_compatible", requested_protocol=None,
        minimum_protocol="coreai-runner.v2", runner_protocol="coreai-runner.v3",
        runner_backward_compatible_with=("coreai-runner.v2",),
        requested_encoding="auto",
        runner_encodings=(NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1))
    assert runtime.protocol_version == base.negotiated_protocol == "coreai-runner.v3"
    assert runtime.observation_encoding == base.negotiated_encoding == NESTED_JSON_V1


def test_auto_picks_first_common_encoding():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1))
    assert negotiate_observation_encoding("auto", caps) == NESTED_JSON_V1


def test_requested_announced_is_used():
    caps = _Caps(observation_encodings=(TYPED_ARRAY_ENVELOPE_V1,))
    assert negotiate_observation_encoding(
        TYPED_ARRAY_ENVELOPE_V1, caps) == TYPED_ARRAY_ENVELOPE_V1


def test_requested_not_announced_fails():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,))
    with pytest.raises(CoreAIPolicyError):
        negotiate_observation_encoding(TYPED_ARRAY_ENVELOPE_V1, caps)


def test_unknown_requested_fails():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,))
    with pytest.raises(CoreAIPolicyError):
        negotiate_observation_encoding("bogus_v9", caps)


def test_no_encodings_without_legacy_fails():
    with pytest.raises(CoreAIPolicyError):
        negotiate_observation_encoding("auto", _Caps(), allow_legacy=False)


def test_no_encodings_with_legacy_warns_and_uses_nested_json():
    with pytest.warns(RuntimeWarning):
        enc = negotiate_observation_encoding("auto", _Caps(), allow_legacy=True)
    assert enc == NESTED_JSON_V1


def test_no_common_encoding_fails():
    caps = _Caps(observation_encodings=("some_future_v9",))
    with pytest.raises(CoreAIPolicyError):
        negotiate_observation_encoding("auto", caps)


# --- full runner-protocol negotiation (v1.3.5) ---

def _v2_caps(**kw):
    return _Caps(observation_encodings=(NESTED_JSON_V1,),
                 protocol_version="coreai-runner.v2", **kw)


def test_protocol_v2_negotiates_and_uses_announced_version():
    neg = negotiate_runner_protocol(
        requested_encoding="auto", capabilities=_v2_caps(supports_batch=True,
                                                         max_batch_size=4))
    assert isinstance(neg, NegotiatedRunnerProtocol)
    assert neg.protocol_version == "coreai-runner.v2"   # announced, not hardcoded
    assert neg.observation_encoding == NESTED_JSON_V1
    assert neg.supports_batch is True and neg.max_batch_size == 4
    assert neg.legacy is False


def test_newer_protocol_without_backward_compat_fails():
    # v1.3.6: a higher major must NOT be accepted blindly (may be breaking).
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,),
                 protocol_version="coreai-runner.v3")
    with pytest.raises(CoreAIPolicyError):
        negotiate_runner_protocol(requested_encoding="auto", capabilities=caps)


def test_newer_protocol_with_backward_compat_is_accepted():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,),
                 protocol_version="coreai-runner.v3",
                 backward_compatible_with=("coreai-runner.v2",))
    neg = negotiate_runner_protocol(requested_encoding="auto", capabilities=caps)
    assert neg.protocol_version == "coreai-runner.v3"


def test_wrong_family_fails():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,),
                 protocol_version="malicious-runner.v2")
    with pytest.raises(CoreAIPolicyError):
        negotiate_runner_protocol(requested_encoding="auto", capabilities=caps)


def test_lower_protocol_fails():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,),
                 protocol_version="coreai-runner.v1")
    with pytest.raises(CoreAIPolicyError):
        negotiate_runner_protocol(requested_encoding="auto", capabilities=caps)


def test_unknown_protocol_fails():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,),
                 protocol_version="something-weird")
    with pytest.raises(CoreAIPolicyError):
        negotiate_runner_protocol(requested_encoding="auto", capabilities=caps)


def test_missing_protocol_without_legacy_fails():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,), protocol_version=None)
    with pytest.raises(CoreAIPolicyError):
        negotiate_runner_protocol(requested_encoding="auto", capabilities=caps)


def test_missing_protocol_with_legacy_warns():
    caps = _Caps(observation_encodings=(NESTED_JSON_V1,), protocol_version=None)
    with pytest.warns(RuntimeWarning):
        neg = negotiate_runner_protocol(
            requested_encoding="auto", capabilities=caps, allow_legacy=True)
    assert neg.legacy is True
    assert neg.protocol_version == "coreai-runner.v2"
