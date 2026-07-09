# test_safety_profiles.py — safety profile loading/validation (v0.9.0).

import json

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.safety_profiles import (
    SafetyProfile,
    load_builtin_profile,
    load_safety_profile,
    profile_from_dict,
    resolve_safety_profile,
)


def _valid_dict(**over):
    d = {
        "schema_version": "lerobot-coreai.safety_profile.v0",
        "name": "test-profile",
        "profile_type": "software_bounds",
        "mode": "fail_closed",
        "max_abs_action": 1.0,
    }
    d.update(over)
    return d


class TestLoad:
    def test_builtin_default_sim_safe(self):
        p = load_builtin_profile("default-sim-safe")
        assert p.name == "default-sim-safe"
        assert p.require_finite is True
        assert p.allow_nan is False
        assert p.source.startswith("builtin:")

    def test_builtin_so100(self):
        p = load_builtin_profile("so100-sim-default")
        assert p.robot_type == "so100"
        assert p.action_shape == [16, 7]
        assert p.max_delta == 0.35
        assert p.max_l2_norm == 6.0
        assert p.profile_type == "software_bounds"

    def test_builtin_so101(self):
        p = load_builtin_profile("so101-sim-default")
        assert p.robot_type == "so101"
        assert p.action_shape == [16, 7]

    def test_builtin_generic_7dof(self):
        p = load_builtin_profile("generic-7dof-sim-default")
        assert p.robot_type is None
        assert p.action_shape == [7]
        assert p.require_robot_type_match is False
        assert p.max_delta == 0.25

    def test_builtin_pusht(self):
        p = load_builtin_profile("pusht-sim-default")
        assert p.action_shape == [2]
        assert p.intended_envs == ["PushT-v0", "pusht"]
        assert p.max_l2_norm == 1.5

    def test_unknown_builtin_fails(self):
        with pytest.raises(CoreAIPolicyError, match="Unknown built-in"):
            load_builtin_profile("does-not-exist")

    def test_load_from_file(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps(_valid_dict(name="from-file")))
        p = load_safety_profile(path)
        assert p.name == "from-file"
        assert p.source == str(path)

    def test_missing_file_fails(self, tmp_path):
        with pytest.raises(CoreAIPolicyError, match="not found"):
            load_safety_profile(tmp_path / "nope.json")

    def test_invalid_json_fails(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text("{ not json")
        with pytest.raises(CoreAIPolicyError, match="Invalid safety profile JSON"):
            load_safety_profile(path)

    def test_non_fail_closed_mode_rejected(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps(_valid_dict(mode="fail_open")))
        with pytest.raises(CoreAIPolicyError):
            load_safety_profile(path)

    def test_missing_schema_version_rejected(self, tmp_path):
        d = _valid_dict()
        del d["schema_version"]
        path = tmp_path / "p.json"
        path.write_text(json.dumps(d))
        with pytest.raises(CoreAIPolicyError):
            load_safety_profile(path)


class TestResolve:
    def test_resolve_path(self, tmp_path):
        path = tmp_path / "p.json"
        path.write_text(json.dumps(_valid_dict(name="explicit")))
        p = resolve_safety_profile(path=path)
        assert p.name == "explicit"

    def test_resolve_name(self):
        p = resolve_safety_profile(name="so100-sim-default")
        assert p.name == "so100-sim-default"

    def test_resolve_default(self):
        p = resolve_safety_profile()
        assert p.name == "default-sim-safe"

    def test_resolve_no_default_fails(self):
        with pytest.raises(CoreAIPolicyError, match="fail-closed"):
            resolve_safety_profile(default_builtin=None)


def test_profile_from_dict_roundtrip():
    p = profile_from_dict(_valid_dict(name="rt"))
    d = p.to_dict()
    assert d["name"] == "rt"
    assert d["schema_version"] == "lerobot-coreai.safety_profile.v0"
