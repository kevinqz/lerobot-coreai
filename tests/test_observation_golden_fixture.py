# test_observation_golden_fixture.py — Python-side golden fixture for the Runner
# observation payload (RFC-0700 §6: "A Swift/Python golden fixture prevents recurrence
# of the observation nesting mismatch"). Pins the EXACT nested serialization + a
# byte-stable digest so any drift in the observation encoding is caught immediately.
# The Swift side must produce the same digest for the same canonical observation.

import numpy as np

from lerobot_coreai.coreai_observation_serialization import (
    serialize_and_hash, serialize_observation,
)

# a canonical multimodal observation (state + front/wrist images + task), exact
# float32 values so the serialization is byte-reproducible.
_CANONICAL = {
    "observation.state": np.array([0.0, 0.5, 1.0, -0.5, 0.25, -1.0], dtype=np.float32),
    "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
    "observation.images.wrist": np.ones((2, 2, 3), dtype=np.uint8),
    "task": "pick the cube",
}

# GOLDEN digest — if this changes, the observation wire format changed. Update it ONLY
# with a deliberate, cross-checked (Swift + Python) protocol change.
_GOLDEN_SHA256 = "sha256:72dd25a5cd960276b75e0287a9e6bc341ccac5474201c724e461049947db8ed8"


def test_observation_nesting_is_exactly_as_expected():
    payload = serialize_observation(_CANONICAL)
    # a tensor becomes a FLAT {"__array__", "dtype", "shape"} node — never double-nested,
    # never a bare list, never a stringified blob.
    st = payload["observation.state"]
    assert set(st) == {"__array__", "dtype", "shape"}
    assert st["__array__"] == [0.0, 0.5, 1.0, -0.5, 0.25, -1.0]
    assert st["dtype"] == "float32" and st["shape"] == [6]
    front = payload["observation.images.front"]
    assert front["dtype"] == "uint8" and front["shape"] == [2, 2, 3]
    assert isinstance(front["__array__"], list)           # nested lists, not a blob
    assert payload["task"] == "pick the cube"             # scalars stay scalar


def test_observation_golden_digest_is_stable():
    _payload, digest = serialize_and_hash(_CANONICAL)
    assert digest == _GOLDEN_SHA256, (
        "observation wire format changed — this is a cross-process (Swift/Python) "
        "protocol break; update the golden only with a deliberate, cross-checked change")
