# test_safety_regression.py — safety regression evaluation (v0.9.2).

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.safety_regression import (
    SafetyRegressionConfig,
    evaluate_safety_regression,
)


def _summary(**over):
    base = {
        "actions_supervised": 1000, "actions_blocked": 0, "actions_modified": 0,
        "would_block_actions": 0, "critical_failures": 0, "critical_findings": 0,
        "passed": True, "profile": "so100-sim-default",
    }
    base.update(over)
    return base


def test_equal_passes():
    r = evaluate_safety_regression(_summary(), _summary(), SafetyRegressionConfig())
    assert r.passed
    assert r.report["claims"]["proves_no_safety_regression_on_compared_artifacts"] is True


def test_more_blocked_fails():
    r = evaluate_safety_regression(
        _summary(), _summary(actions_blocked=3, passed=False), SafetyRegressionConfig())
    assert not r.passed
    assert any(c.name == "max_blocked_increase" and not c.passed for c in r.checks)
    assert r.report["claims"]["proves_no_safety_regression_on_compared_artifacts"] is False


def test_higher_block_rate_fails():
    base = _summary(actions_supervised=1000, actions_blocked=0)
    cand = _summary(actions_supervised=500, actions_blocked=2, passed=False)
    r = evaluate_safety_regression(base, cand, SafetyRegressionConfig())
    assert any(c.name == "max_block_rate_increase" and not c.passed for c in r.checks)


def test_more_critical_findings_fails():
    r = evaluate_safety_regression(
        _summary(), _summary(critical_findings=2, passed=False), SafetyRegressionConfig())
    assert any(c.name == "max_critical_findings_increase" and not c.passed for c in r.checks)


def test_fewer_actions_warns():
    r = evaluate_safety_regression(
        _summary(actions_supervised=1000), _summary(actions_supervised=200),
        SafetyRegressionConfig())
    assert any("fewer actions" in w for w in r.warnings)


def test_require_same_profile_fails_on_mismatch():
    r = evaluate_safety_regression(
        _summary(profile="so100-sim-default"), _summary(profile="so101-sim-default"),
        SafetyRegressionConfig(require_same_profile=True))
    assert any(c.name == "require_same_profile" and not c.passed for c in r.checks)


def test_require_candidate_passed():
    r = evaluate_safety_regression(
        _summary(), _summary(passed=False), SafetyRegressionConfig())
    assert any(c.name == "require_candidate_passed" and not c.passed for c in r.checks)


def test_malformed_summary_fails_closed():
    with pytest.raises(CoreAIPolicyError):
        evaluate_safety_regression({}, _summary(), SafetyRegressionConfig())
    with pytest.raises(CoreAIPolicyError):
        evaluate_safety_regression(_summary(), {}, SafetyRegressionConfig())


def test_physical_safety_claims_always_false():
    r = evaluate_safety_regression(_summary(), _summary(), SafetyRegressionConfig())
    assert r.report["claims"]["proves_physical_safety"] is False
    assert r.report["claims"]["proves_real_world_safety"] is False


def test_candidate_improvement_passes():
    # Candidate blocks fewer → negative delta → passes.
    r = evaluate_safety_regression(
        _summary(actions_blocked=5, passed=False), _summary(actions_blocked=0),
        SafetyRegressionConfig(require_candidate_passed=True))
    assert r.passed
