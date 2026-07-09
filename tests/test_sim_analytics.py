# test_sim_analytics.py — unit tests for sim analytics aggregation (v0.8.2).

import json
import pytest
from pathlib import Path

from lerobot_coreai.sim_analytics import (
    aggregate_episode_metrics,
    aggregate_step_metrics,
    aggregate_failure_metrics,
    build_sim_analytics,
    load_jsonl,
    percentile,
    mean,
    median,
    write_episode_metrics_csv,
    write_step_metrics_csv,
)


class TestStatsHelpers:
    def test_percentile_empty(self):
        assert percentile([], 50) is None

    def test_percentile_single(self):
        assert percentile([5.0], 50) == 5.0

    def test_percentile_median(self):
        assert percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_percentile_p95(self):
        # 0..99, p95 interpolates to 94.05
        vals = list(range(100))
        assert percentile(vals, 95) == pytest.approx(94.05)

    def test_mean_empty(self):
        assert mean([]) is None

    def test_median_empty(self):
        assert median([]) is None


class TestEpisodeMetrics:
    def test_empty_episodes(self):
        m = aggregate_episode_metrics([])
        assert m["episodes_completed"] == 0
        assert m["success_rate"] is None
        assert m["mean_reward"] is None
        assert m["median_reward"] is None

    def test_success_rate_and_rewards(self):
        episodes = [
            {"episode": 0, "steps": 10, "total_reward": 5.0, "success": True, "terminated": True, "truncated": False},
            {"episode": 1, "steps": 20, "total_reward": 1.0, "success": False, "terminated": False, "truncated": True},
        ]
        m = aggregate_episode_metrics(episodes)
        assert m["episodes_completed"] == 2
        assert m["success_rate"] == 0.5
        assert m["mean_reward"] == 3.0
        assert m["min_reward"] == 1.0
        assert m["max_reward"] == 5.0
        assert m["mean_steps"] == 15.0
        assert m["terminated_episodes"] == 1
        assert m["truncated_episodes"] == 1


class TestStepMetrics:
    def _action_records(self):
        return [
            {"ok": True, "timing": {"runner_total_ms": 10.0, "loop_total_ms": 15.0, "env_step_ms": 2.0},
             "diagnostics": {"mean_abs": 0.01, "max_abs": 0.1, "nan_count": 0, "inf_count": 0},
             "action_shape": [16, 7]},
            {"ok": True, "timing": {"runner_total_ms": 20.0, "loop_total_ms": 25.0, "env_step_ms": 4.0},
             "diagnostics": {"mean_abs": 0.05, "max_abs": 0.2, "nan_count": 1, "inf_count": 0},
             "action_shape": [16, 7]},
        ]

    def test_latency_p50_p95(self):
        latency, _ = aggregate_step_metrics(self._action_records())
        assert latency["runner_p50_ms"] == 15.0
        assert latency["env_step_p50_ms"] == 3.0
        assert latency["loop_max_ms"] == 25.0

    def test_action_metrics(self):
        _, action = aggregate_step_metrics(self._action_records())
        assert action["mean_abs_action"] == pytest.approx(0.03)
        assert action["max_abs_action"] == 0.2
        assert action["nan_action_steps"] == 1
        assert action["inf_action_steps"] == 0
        assert action["unique_action_shapes"] == [[16, 7]]
        assert action["shape_changes"] == 0

    def test_shape_changes_detected(self):
        records = self._action_records()
        records[1]["action_shape"] = [8, 7]
        _, action = aggregate_step_metrics(records)
        assert action["shape_changes"] == 1


class TestFailureMetrics:
    def test_no_errors(self):
        m = aggregate_failure_metrics([], episodes_completed=5, total_steps=50)
        assert m["total_errors"] == 0
        assert m["error_rate"] == 0.0

    def test_classified_errors(self):
        errors = [
            {"stage": "action.generate", "type": "RunnerTimeoutError", "message": "x"},
            {"stage": "simulator.step", "type": "CoreAIPolicyError", "message": "y"},
            {"stage": "observation.adapt", "type": "ObservationValidationError", "message": "z"},
        ]
        m = aggregate_failure_metrics(errors, episodes_completed=3, total_steps=30)
        assert m["total_errors"] == 3
        assert m["runner_errors"] == 1
        assert m["env_errors"] == 1
        assert m["error_rate"] == pytest.approx(0.1)


class TestBuildSimAnalytics:
    def test_build_from_jsonl(self, tmp_path):
        actions_path = tmp_path / "actions.jsonl"
        episodes_path = tmp_path / "episodes.jsonl"
        actions_path.write_text("\n".join(json.dumps(r) for r in [
            {"ok": True, "timing": {"runner_total_ms": 10.0, "loop_total_ms": 15.0, "env_step_ms": 2.0},
             "diagnostics": {"mean_abs": 0.01, "max_abs": 0.1, "nan_count": 0, "inf_count": 0},
             "action_shape": [16, 7]},
        ]))
        episodes_path.write_text(json.dumps(
            {"episode": 0, "steps": 1, "total_reward": 1.0, "success": True,
             "terminated": True, "truncated": False}
        ))
        analytics = build_sim_analytics(actions_path=actions_path, episodes_path=episodes_path, errors=[])
        assert "episode_metrics" in analytics
        assert "latency_metrics" in analytics
        assert "action_metrics" in analytics
        assert "failure_metrics" in analytics
        assert analytics["episode_metrics"]["episodes_completed"] == 1


class TestCsvExports:
    def test_episode_csv(self, tmp_path):
        episodes = [{"episode": 0, "steps": 10, "total_reward": 5.0, "success": True,
                     "terminated": True, "truncated": False,
                     "actions_sent_to_simulator": 10, "actions_sent_to_robot": 0}]
        path = tmp_path / "episode_metrics.csv"
        write_episode_metrics_csv(path, episodes)
        content = path.read_text()
        assert "episode,steps,total_reward" in content
        assert "0,10,5.0" in content

    def test_step_csv(self, tmp_path):
        actions = [{"episode": 0, "step": 0, "ok": True, "reward": 1.0, "done": False,
                    "timing": {"runner_total_ms": 10.0, "env_step_ms": 2.0, "loop_total_ms": 15.0},
                    "diagnostics": {"mean_abs": 0.01, "max_abs": 0.1, "nan_count": 0, "inf_count": 0},
                    "action_shape": [16, 7], "error": None}]
        path = tmp_path / "step_metrics.csv"
        write_step_metrics_csv(path, actions)
        content = path.read_text()
        assert "episode,step,ok,reward" in content
        assert "[16, 7]" in content

    def test_load_jsonl_missing_file(self, tmp_path):
        assert load_jsonl(tmp_path / "nonexistent.jsonl") == []
