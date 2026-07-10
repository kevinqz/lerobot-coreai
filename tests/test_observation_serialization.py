# test_observation_serialization.py — JSON-safe observation boundary (v1.2.8).

import json

import pytest

from lerobot_coreai.coreai_observation_serialization import (
    observation_sha256, serialize_and_hash, serialize_observation, serialize_value,
)
from lerobot_coreai.errors import CoreAIPolicyError


def test_primitives_and_nested_pass_through():
    obs = {"observation.state": [0.0, 1.0], "task": "do", "n": 3, "flag": True,
           "nested": {"a": [1, 2]}}
    out = serialize_observation(obs)
    assert out == obs
    # Must be JSON-serializable.
    json.dumps(out)


def test_unknown_object_rejected():
    class Weird:
        pass
    with pytest.raises(CoreAIPolicyError):
        serialize_observation({"x": Weird()})


def test_hash_is_deterministic():
    obs = {"b": 2, "a": 1}
    _, h1 = serialize_and_hash(obs)
    _, h2 = serialize_and_hash({"a": 1, "b": 2})
    assert h1 == h2 and h1.startswith("sha256:")


def test_numpy_array_converted_with_metadata():
    np = pytest.importorskip("numpy")
    arr = np.zeros((2, 3), dtype=np.float32)
    out = serialize_value(arr)
    assert out["__array__"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert out["shape"] == [2, 3]
    assert "float32" in out["dtype"]
    json.dumps(out)


def test_observation_with_array_is_json_safe():
    np = pytest.importorskip("numpy")
    obs = {"observation.state": np.arange(3, dtype=np.float32), "task": "t"}
    payload, h = serialize_and_hash(obs)
    json.dumps(payload)
    assert h.startswith("sha256:")
