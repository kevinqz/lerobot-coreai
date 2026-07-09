# test_sim_quality.py — tests for sim quality gates (v0.8.3).

import pytest

from lerobot_coreai.sim_quality import SimQualityConfig, evaluate_sim_quality


def _analytics(**overrides):
    base = {
        "episode_metrics": {"success_rate": 0.9, "mean_reward": 42.0},
        "latency_metrics": {"runner_p95_ms": 10.0, "env_step_p95_ms": 5.0, "loop_p95_ms": 20.0},
        "action_metrics": {"nan_action_steps": 0, "inf_action_steps": 0, "shape_changes": 0},
        "failure_metrics": {"error_rate": 0.0},
    }
    base.update(overrides)
    return base


class TestSimQuality:
    def test_all_pass(self):
        r = evaluate_sim_quality(_analytics(), SimQualityConfig(), error_rate=0.0)
        assert r.passed is True

    def test_fail_min_success_rate(self):
        r = evaluate_sim_quality(
            _analytics(episode_metrics={"success_rate": 0.6, "mean_reward": 42.0}),
            SimQualityConfig(min_success_rate=0.8),
            error_rate=0.0,
        )
        assert r.passed is False
        check = [c for c in r.checks if c["name"] == "min_success_rate"][0]
        assert check["passed"] is False

    def test_fail_min_mean_reward(self):
        r = evaluate_sim_quality(
            _analytics(episode_metrics={"success_rate": 0.9, "mean_reward": 5.0}),
            SimQualityConfig(min_mean_reward=10.0),
            error_rate=0.0,
        )
        assert r.passed is False

    def test_fail_runner_latency(self):
        r = evaluate_sim_quality(
            _analytics(),
            SimQualityConfig(max_runner_p95_ms=8.0),
            error_rate=0.0,
        )
        assert r.passed is False
        check = [c for c in r.checks if c["name"] == "max_runner_p95_ms"][0]
        assert check["passed"] is False

    def test_fail_env_step_latency(self):
        r = evaluate_sim_quality(
            _analytics(),
            SimQualityConfig(max_env_step_p95_ms=3.0),
            error_rate=0.0,
        )
        assert r.passed is False

    def test_fail_error_rate(self):
        r = evaluate_sim_quality(
            _analytics(), SimQualityConfig(max_error_rate=0.05), error_rate=0.2,
        )
        assert r.passed is False

    def test_fail_nan_actions(self):
        r = evaluate_sim_quality(
            _analytics(action_metrics={"nan_action_steps": 2, "inf_action_steps": 0, "shape_changes": 0}),
            SimQualityConfig(),
            error_rate=0.0,
        )
        assert r.passed is False

    def test_fail_shape_changes(self):
        r = evaluate_sim_quality(
            _analytics(action_metrics={"nan_action_steps": 0, "inf_action_steps": 0, "shape_changes": 1}),
            SimQualityConfig(),
            error_rate=0.0,
        )
        assert r.passed is False

    def test_allow_shape_changes(self):
        r = evaluate_sim_quality(
            _analytics(action_metrics={"nan_action_steps": 0, "inf_action_steps": 0, "shape_changes": 1}),
            SimQualityConfig(allow_action_shape_changes=True),
            error_rate=0.0,
        )
        # shape change check is skipped when allowed
        assert not any(c["name"] == "no_action_shape_changes" for c in r.checks)
        assert r.passed is True

    def test_disabled_checks_pass(self):
        r = evaluate_sim_quality(_analytics(), SimQualityConfig(), error_rate=0.0)
        # With no thresholds set beyond defaults, should pass with default checks.
        assert r.passed is True
