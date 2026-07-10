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


# --- v1.3.12: enum + alias hardening (via the real capabilities parser) ---

import httpx  # noqa: E402
from lerobot_coreai.runner import RunnerClient  # noqa: E402


def _client_with(caps: dict) -> RunnerClient:
    def handler(request):
        if request.url.path.endswith("/capabilities"):
            return httpx.Response(200, json=caps)
        return httpx.Response(200, json={"status": "ok"})
    c = RunnerClient("http://runner")
    c._client = httpx.Client(transport=httpx.MockTransport(handler),
                             base_url="http://runner")
    return c


def _base_caps(**batching):
    ab = {"supported": True, "max_batch_size": 4, "semantics": "native"}
    ab.update(batching)
    return {"runtime": "coreai-runner", "supports": {"action": True},
            "protocol_version": "coreai-runner.v2",
            "observation_encodings": ["nested_json_v1"], "action_batching": ab,
            "inference_state": {"scope": "stateless"}}


def test_unknown_semantics_fails():
    with pytest.raises(RunnerProtocolError):
        _client_with(_base_caps(semantics="turbo")).capabilities()


def test_unknown_slot_isolation_fails():
    with pytest.raises(RunnerProtocolError):
        _client_with(_base_caps(slot_isolation="mystery")).capabilities()


def test_conflicting_isolation_aliases_fail():
    with pytest.raises(RunnerProtocolError):
        _client_with(_base_caps(slot_isolation="independent",
                                state_isolation="shared")).capabilities()


def test_string_supported_fails():
    with pytest.raises(RunnerProtocolError):
        _client_with(_base_caps(supported="false")).capabilities()


def test_encodings_must_be_list():
    caps = _base_caps()
    caps["observation_encodings"] = "nested_json_v1"
    with pytest.raises(RunnerProtocolError):
        _client_with(caps).capabilities()


def test_valid_caps_parse_ok():
    caps = _client_with(_base_caps(slot_isolation="independent")).capabilities()
    assert caps.action_batching_slot_isolation == "independent"
    assert caps.action_batching_semantics == "native"
