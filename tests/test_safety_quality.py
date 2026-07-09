# test_safety_quality.py — safety quality gate evaluation (v0.9.2).

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.safety_quality import (
    SafetyQualityConfig,
    build_safety_quality_report,
    evaluate_safety_quality,
)


def test_malformed_summary_missing_counts_fails_closed():
    with pytest.raises(CoreAIPolicyError, match="Malformed safety summary"):
        evaluate_safety_quality({"passed": True}, SafetyQualityConfig())


def test_noninteger_count_field_fails_closed():
    with pytest.raises(CoreAIPolicyError, match="actions_blocked must be"):
        evaluate_safety_quality({
            "actions_supervised": 10, "actions_blocked": "two", "actions_modified": 0,
            "critical_failures": 0, "critical_findings": 0, "would_block_actions": 0,
            "passed": True,
        }, SafetyQualityConfig())


def test_nonbool_passed_fails_closed():
    with pytest.raises(CoreAIPolicyError, match="passed must be a boolean"):
        evaluate_safety_quality({
            "actions_supervised": 10, "actions_blocked": 0, "actions_modified": 0,
            "critical_failures": 0, "critical_findings": 0, "would_block_actions": 0,
            "passed": "yes",
        }, SafetyQualityConfig())


def test_zero_actions_fails_closed():
    with pytest.raises(CoreAIPolicyError, match="actions_supervised"):
        evaluate_safety_quality({
            "actions_supervised": 0, "actions_blocked": 0, "actions_modified": 0,
            "critical_failures": 0, "critical_findings": 0, "would_block_actions": 0,
            "passed": True,
        }, SafetyQualityConfig())


def _summary(**over):
    base = {
        "actions_supervised": 1000, "actions_allowed": 1000, "actions_blocked": 0,
        "actions_modified": 0, "critical_failures": 0, "would_block_actions": 0,
        "critical_findings": 0, "top_reasons": {}, "passed": True,
    }
    base.update(over)
    return base


def test_clean_summary_passes_defaults():
    r = evaluate_safety_quality(_summary(), SafetyQualityConfig())
    assert r.passed


def test_blocked_action_fails():
    r = evaluate_safety_quality(
        _summary(actions_blocked=2, passed=False), SafetyQualityConfig())
    assert not r.passed
    assert any(c.name == "max_actions_blocked" and not c.passed for c in r.checks)


def test_block_rate_fails():
    r = evaluate_safety_quality(
        _summary(actions_blocked=5, passed=False), SafetyQualityConfig())
    c = next(c for c in r.checks if c.name == "max_block_rate")
    assert not c.passed
    assert c.value == pytest.approx(0.005)


def test_critical_findings_fail():
    r = evaluate_safety_quality(
        _summary(critical_findings=3, passed=False), SafetyQualityConfig())
    assert any(c.name == "max_critical_findings" and not c.passed for c in r.checks)


def test_would_block_fails():
    r = evaluate_safety_quality(
        _summary(would_block_actions=1, passed=False), SafetyQualityConfig())
    assert any(c.name == "max_would_block_actions" and not c.passed for c in r.checks)


def test_modification_rate_check_optional():
    # Not configured by default → no modification-rate check present.
    r = evaluate_safety_quality(_summary(actions_modified=500), SafetyQualityConfig())
    assert not any(c.name == "max_modification_rate" for c in r.checks)
    # Configured → enforced.
    r2 = evaluate_safety_quality(
        _summary(actions_modified=500), SafetyQualityConfig(max_modification_rate=0.1))
    assert any(c.name == "max_modification_rate" and not c.passed for c in r2.checks)


def test_parse_errors_fail():
    r = evaluate_safety_quality(
        _summary(top_reasons={"unparseable_actions_line": 2}, passed=False),
        SafetyQualityConfig())
    assert any(c.name == "require_zero_parse_errors" and not c.passed for c in r.checks)


def test_allow_parse_errors():
    r = evaluate_safety_quality(
        _summary(top_reasons={"unparseable_actions_line": 2}),
        SafetyQualityConfig(require_zero_parse_errors=False))
    assert not any(c.name == "require_zero_parse_errors" for c in r.checks)


def test_min_actions_supervised_fails():
    r = evaluate_safety_quality(
        _summary(actions_supervised=5), SafetyQualityConfig(min_actions_supervised=10))
    assert any(c.name == "min_actions_supervised" and not c.passed for c in r.checks)


def test_require_passed_summary():
    r = evaluate_safety_quality(_summary(passed=False), SafetyQualityConfig())
    assert any(c.name == "require_passed_summary" and not c.passed for c in r.checks)
    # Disabled.
    r2 = evaluate_safety_quality(
        _summary(passed=False), SafetyQualityConfig(require_passed_summary=False))
    assert not any(c.name == "require_passed_summary" for c in r2.checks)


def test_delta_and_shape_and_nonfinite_from_reasons():
    r = evaluate_safety_quality(
        _summary(actions_blocked=3, passed=False, critical_findings=3, top_reasons={
            "delta": 1, "shape": 1, "finite": 1}),
        SafetyQualityConfig())
    names = {c.name: c for c in r.checks}
    assert not names["max_delta_failures"].passed
    assert not names["max_shape_failures"].passed
    assert not names["max_nonfinite_failures"].passed


def test_report_claims_honest():
    r = evaluate_safety_quality(_summary(actions_blocked=1, passed=False), SafetyQualityConfig())
    report = build_safety_quality_report(r, source={"type": "safety_summary", "path": "x"})
    assert report["claims"]["proves_software_safety_quality"] is True
    assert report["claims"]["proves_physical_safety"] is False
    assert report["claims"]["proves_real_world_safety"] is False
    assert report["passed"] is False
