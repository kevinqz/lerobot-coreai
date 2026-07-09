# test_builtin_profiles.py — built-in safety profile correctness (v0.9.1).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.safety_profiles import list_builtin_profiles, load_builtin_profile

EXPECTED = {"default-sim-safe", "generic-7dof-sim-default", "so100-sim-default",
            "so101-sim-default", "pusht-sim-default"}


def _schema():
    return json.loads(
        files("lerobot_coreai.schemas").joinpath("safety-profile.schema.json").read_text())


def test_list_contains_all_expected():
    assert EXPECTED.issubset(set(list_builtin_profiles()))


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_builtin_loads_and_validates(name):
    p = load_builtin_profile(name)
    assert p.mode == "fail_closed"
    assert p.profile_type == "software_bounds"
    assert p.require_finite is True
    # Every built-in must acknowledge it does not prove physical safety.
    lim = " ".join(p.limitations).lower()
    assert "physical" in lim or "real-world" in lim
    # And validate against the schema.
    data = json.loads(
        files("lerobot_coreai.profiles").joinpath(f"{name}.json").read_text())
    jsonschema.validate(data, _schema())


def test_robot_types():
    assert load_builtin_profile("so100-sim-default").robot_type == "so100"
    assert load_builtin_profile("so101-sim-default").robot_type == "so101"
    assert load_builtin_profile("pusht-sim-default").robot_type is None


def test_shapes():
    assert load_builtin_profile("so100-sim-default").action_shape == [16, 7]
    assert load_builtin_profile("generic-7dof-sim-default").action_shape == [7]
    assert load_builtin_profile("pusht-sim-default").action_shape == [2]


def test_no_builtin_overclaims_physical_safety():
    for name in EXPECTED:
        p = load_builtin_profile(name)
        blob = json.dumps(p.to_dict()).lower()
        assert "proves_physical_safety" not in blob  # profiles carry no such claim
