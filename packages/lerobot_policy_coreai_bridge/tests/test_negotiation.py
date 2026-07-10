# test_negotiation.py — real observation-encoding negotiation (v1.3.4).

from dataclasses import dataclass

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_policy_coreai_bridge.negotiation import negotiate_observation_encoding
from lerobot_policy_coreai_bridge.transport import (
    NESTED_JSON_V1, TYPED_ARRAY_ENVELOPE_V1,
)


@dataclass
class _Caps:
    observation_encodings: tuple = ()


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
