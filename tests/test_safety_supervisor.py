# test_safety_supervisor.py — runtime safety supervisor unit tests (v0.9.0).

import math

import pytest

from lerobot_coreai.safety_profiles import SafetyProfile
from lerobot_coreai.safety_supervisor import (
    MODE_ENFORCE,
    MODE_REPORT_ONLY,
    SafetyContext,
    SafetySupervisor,
    normalize_action,
    reshape,
    safe_evaluate,
)


def _profile(**over) -> SafetyProfile:
    base = dict(name="t", require_robot_type_match=False, require_known_shape=False)
    base.update(over)
    return SafetyProfile(**base)


def _sup(mode=MODE_ENFORCE, **over) -> SafetySupervisor:
    return SafetySupervisor(_profile(**over), mode=mode)


class TestNormalize:
    def test_flat_and_shape(self):
        flat, shape = normalize_action([[1.0, 2.0], [3.0, 4.0]])
        assert flat == [1.0, 2.0, 3.0, 4.0]
        assert shape == [2, 2]

    def test_dict_with_action_key(self):
        flat, shape = normalize_action({"action": [1.0, 2.0], "metadata": {"x": 1}})
        assert flat == [1.0, 2.0]
        assert shape == [2]

    def test_none(self):
        assert normalize_action(None) == (None, None)

    def test_reshape_roundtrip(self):
        assert reshape([1, 2, 3, 4], [2, 2]) == [[1, 2], [3, 4]]
        assert reshape([1, 2, 3], [3]) == [1, 2, 3]


class TestBasicChecks:
    def test_allows_valid_bounded_action(self):
        r = _sup(min_action=-1.0, max_action=1.0).evaluate([[0.1] * 7] * 16)
        assert r.decision.allowed
        assert r.action is not None
        assert r.decision.severity == "info"

    def test_blocks_missing_action(self):
        r = _sup().evaluate(None)
        assert not r.decision.allowed
        assert r.action is None
        assert "action_present" in r.decision.reasons

    def test_blocks_nan(self):
        r = _sup().evaluate([0.0, float("nan"), 0.0])
        assert not r.decision.allowed
        assert "finite" in r.decision.reasons

    def test_blocks_inf(self):
        r = _sup().evaluate([0.0, float("inf")])
        assert not r.decision.allowed
        assert "finite" in r.decision.reasons

    def test_allows_nan_when_profile_permits(self):
        r = _sup(allow_nan=True, allow_inf=True).evaluate([float("nan")])
        assert r.decision.allowed

    def test_blocks_shape_mismatch(self):
        r = _sup(require_known_shape=True, action_shape=[16, 7]).evaluate([[0.0] * 7] * 8)
        assert not r.decision.allowed
        assert "shape" in r.decision.reasons

    def test_allows_shape_change_when_permitted(self):
        r = _sup(action_shape=[16, 7], allow_shape_change=True).evaluate([[0.0] * 7] * 8)
        assert r.decision.allowed

    def test_blocks_unknown_shape_when_required(self):
        # A ragged action → shape None → blocked when require_known_shape.
        r = _sup(require_known_shape=True).evaluate([[0.0, 0.0], [0.0]])
        assert not r.decision.allowed
        assert "shape" in r.decision.reasons


class TestBounds:
    def test_blocks_bounds_exceeded_when_clip_disabled(self):
        r = _sup(max_abs_action=1.0, clip_to_bounds=False).evaluate([5.0, 0.0])
        assert not r.decision.allowed
        assert "bounds" in r.decision.reasons

    def test_clips_bounds_when_enabled(self):
        r = _sup(max_abs_action=1.0, clip_to_bounds=True).evaluate([5.0, -3.0, 0.5])
        assert r.decision.allowed
        assert r.decision.action_modified
        assert "action_clipped_to_bounds" in r.decision.reasons
        assert r.action == [1.0, -1.0, 0.5]

    def test_block_on_clip(self):
        r = _sup(max_abs_action=1.0, clip_to_bounds=True, block_on_clip=True).evaluate([5.0])
        assert not r.decision.allowed
        assert "blocked_due_to_clip" in r.decision.reasons

    def test_min_max_bounds(self):
        r = _sup(min_action=0.0, max_action=2.0, clip_to_bounds=True).evaluate([-1.0, 3.0, 1.0])
        assert r.action == [0.0, 2.0, 1.0]

    def test_max_abs_exceeded_blocks(self):
        r = _sup(max_abs_action=2.0).evaluate([2.5])
        assert not r.decision.allowed


class TestDeltaAndNorm:
    def test_first_action_skips_delta(self):
        sup = _sup(max_delta=0.1)
        r = sup.evaluate([10.0, 10.0])  # large but no previous
        assert r.decision.allowed
        delta = next(c for c in r.decision.checks if c["name"] == "delta")
        assert delta["reason"] == "first_action_no_previous"

    def test_delta_exceeded_blocks(self):
        sup = _sup(max_delta=0.5)
        sup.evaluate([0.0, 0.0])           # establishes previous
        r = sup.evaluate([1.0, 0.0])        # delta 1.0 > 0.5
        assert not r.decision.allowed
        assert "delta" in r.decision.reasons

    def test_delta_within_bound_allowed(self):
        sup = _sup(max_delta=0.5)
        sup.evaluate([0.0, 0.0])
        r = sup.evaluate([0.2, -0.1])
        assert r.decision.allowed

    def test_l2_norm_exceeded_blocks(self):
        r = _sup(max_l2_norm=1.0).evaluate([1.0, 1.0, 1.0])  # norm ~1.73
        assert not r.decision.allowed
        assert "l2_norm" in r.decision.reasons

    def test_l2_norm_within_allowed(self):
        r = _sup(max_l2_norm=2.0).evaluate([1.0, 1.0])  # norm ~1.41
        assert r.decision.allowed

    def test_delta_unverifiable_on_shape_change_blocks(self):
        # v0.9.1: with max_delta set and allow_shape_change, a length change
        # cannot verify the delta bound → fail-closed (previously it passed).
        sup = _sup(max_delta=0.5, allow_shape_change=True)
        sup.evaluate([0.0, 0.0])            # establishes previous (len 2)
        r = sup.evaluate([0.0, 0.0, 0.0])    # len 3 → unverifiable
        assert not r.decision.allowed
        delta = next(c for c in r.decision.checks if c["name"] == "delta")
        assert delta["reason"] == "delta_unverifiable_shape_changed"

    def test_shape_change_allowed_without_delta_bound(self):
        # No max_delta → shape changes are permitted (generic-7dof style).
        sup = _sup(allow_shape_change=True)
        sup.evaluate([0.0, 0.0])
        r = sup.evaluate([0.0, 0.0, 0.0])
        assert r.decision.allowed


class TestRobotType:
    def test_robot_type_mismatch_blocks(self):
        sup = _sup(require_robot_type_match=True, robot_type="so100")
        r = sup.evaluate([0.0], context=SafetyContext(mode="sim", robot_type="so101"))
        assert not r.decision.allowed
        assert "robot_type" in r.decision.reasons

    def test_robot_type_match_allows(self):
        sup = _sup(require_robot_type_match=True, robot_type="so100")
        r = sup.evaluate([0.0], context=SafetyContext(mode="sim", robot_type="so100"))
        assert r.decision.allowed


class TestModes:
    def test_report_only_records_failure_but_returns_action(self):
        sup = _sup(mode=MODE_REPORT_ONLY)
        r = sup.evaluate([float("nan")])
        assert r.decision.allowed            # report-only never blocks
        assert r.action is not None          # original action returned
        assert "report_only_would_block" in r.decision.reasons

    def test_report_only_does_not_modify(self):
        sup = _sup(mode=MODE_REPORT_ONLY, max_abs_action=1.0, clip_to_bounds=True)
        r = sup.evaluate([5.0])
        assert r.action == [5.0]             # not clipped
        assert r.decision.action_modified is False

    def test_enforce_blocks_failure(self):
        sup = _sup(mode=MODE_ENFORCE)
        r = sup.evaluate([float("inf")])
        assert not r.decision.allowed


class TestFailClosed:
    def test_internal_error_fail_closed(self):
        class Boom(SafetySupervisor):
            def evaluate(self, action, *, context=None):
                raise RuntimeError("boom")
        sup = Boom(_profile())
        r = safe_evaluate(sup, [0.0])
        assert not r.decision.allowed
        assert r.action is None
        assert "supervisor_internal_error" in r.decision.reasons
        assert r.decision.severity == "critical"

    def test_safe_evaluate_passthrough_when_ok(self):
        r = safe_evaluate(_sup(), [0.0, 0.0])
        assert r.decision.allowed
