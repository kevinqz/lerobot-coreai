# test_shadow_quality.py — tests for shadow run quality gates (v0.7.2).

import pytest

from lerobot_coreai.shadow_quality import (
    ShadowQualityConfig,
    ShadowQualityResult,
    evaluate_shadow_quality,
)


def _summary(**overrides):
    base = {
        "samples": 32,
        "mean_loop_ms": 15.8,
        "p50_loop_ms": 14.9,
        "p95_loop_ms": 20.1,
        "max_loop_ms": 35.4,
        "mean_runner_ms": 12.3,
        "p95_runner_ms": 15.2,
        "processing_fps": 63.3,
        "effective_fps": 9.7,
        "latency_spikes": 0,
        "nan_actions": 0,
        "inf_actions": 0,
        "shape_changes": 0,
    }
    base.update(overrides)
    return base


class TestShadowQuality:
    def test_all_pass(self):
        config = ShadowQualityConfig()
        result = evaluate_shadow_quality(_summary(), config)
        assert result.passed is True

    def test_runner_p95_fail(self):
        config = ShadowQualityConfig(max_runner_p95_ms=10.0)
        result = evaluate_shadow_quality(_summary(p95_runner_ms=15.2), config)
        assert result.passed is False
        runner_check = [c for c in result.checks if c["name"] == "max_runner_p95_ms"][0]
        assert runner_check["passed"] is False

    def test_loop_p95_fail(self):
        config = ShadowQualityConfig(max_loop_p95_ms=15.0)
        result = evaluate_shadow_quality(_summary(p95_loop_ms=20.1), config)
        assert result.passed is False

    def test_min_fps_fail(self):
        config = ShadowQualityConfig(min_effective_fps=15.0)
        result = evaluate_shadow_quality(_summary(effective_fps=9.7), config)
        assert result.passed is False

    def test_error_rate_fail(self):
        config = ShadowQualityConfig(max_error_rate=0.0)
        result = evaluate_shadow_quality(_summary(), config, error_rate=0.1)
        assert result.passed is False

    def test_nan_actions_fail(self):
        config = ShadowQualityConfig(max_nan_actions=0)
        result = evaluate_shadow_quality(_summary(nan_actions=3), config)
        assert result.passed is False

    def test_inf_actions_fail(self):
        config = ShadowQualityConfig(max_inf_actions=0)
        result = evaluate_shadow_quality(_summary(inf_actions=1), config)
        assert result.passed is False

    def test_shape_changes_fail(self):
        config = ShadowQualityConfig(allow_action_shape_changes=False)
        result = evaluate_shadow_quality(_summary(shape_changes=1), config)
        assert result.passed is False

    def test_shape_changes_allowed(self):
        config = ShadowQualityConfig(allow_action_shape_changes=True)
        result = evaluate_shadow_quality(_summary(shape_changes=5), config)
        # Should not have a shape change check when allowed
        assert not any(c["name"] == "no_action_shape_changes" for c in result.checks)

    def test_checks_have_details(self):
        config = ShadowQualityConfig(max_runner_p95_ms=50.0)
        result = evaluate_shadow_quality(_summary(p95_runner_ms=15.2), config)
        runner_check = [c for c in result.checks if c["name"] == "max_runner_p95_ms"][0]
        assert runner_check["value"] == 15.2
        assert runner_check["threshold"] == 50.0
        assert runner_check["passed"] is True
