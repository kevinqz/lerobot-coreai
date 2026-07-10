# test_runner_capabilities_strict.py — strict capability parsing + fingerprint (v1.3.10).

import pytest

from lerobot_coreai.errors import RunnerProtocolError
from lerobot_coreai.runner import _strict_bool, capabilities_sha256
from lerobot_coreai.types import RunnerCapabilities


def test_strict_bool_accepts_real_bools():
    assert _strict_bool(True, "x") is True
    assert _strict_bool(False, "x") is False
    assert _strict_bool(None, "x") is False   # absent -> False


def test_strict_bool_rejects_string_false():
    # bool("false") is True in Python — must NOT be coerced.
    with pytest.raises(RunnerProtocolError):
        _strict_bool("false", "action_batching.supported")


def test_strict_bool_rejects_int():
    with pytest.raises(RunnerProtocolError):
        _strict_bool(1, "x")


def test_capabilities_fingerprint_is_stable_and_order_insensitive():
    a = RunnerCapabilities(raw={"protocol_version": "coreai-runner.v2",
                                "action_batching": {"supported": True, "max_batch_size": 4}})
    b = RunnerCapabilities(raw={"action_batching": {"max_batch_size": 4, "supported": True},
                                "protocol_version": "coreai-runner.v2"})
    assert capabilities_sha256(a) == capabilities_sha256(b)
    assert capabilities_sha256(a).startswith("sha256:")


def test_capabilities_fingerprint_differs_on_change():
    a = RunnerCapabilities(raw={"action_batching": {"max_batch_size": 4}})
    b = RunnerCapabilities(raw={"action_batching": {"max_batch_size": 8}})
    assert capabilities_sha256(a) != capabilities_sha256(b)
