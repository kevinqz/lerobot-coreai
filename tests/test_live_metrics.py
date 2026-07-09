# test_live_metrics.py — tests for live metrics collection (v0.7.2).

import math
import pytest

from lerobot_coreai.live_metrics import (
    LiveMetricSample,
    LiveMetricsCollector,
    summarize_action,
)


class TestSummarizeAction:
    def test_basic_action(self):
        result = summarize_action([[0.1, 0.2], [0.3, 0.4]])
        assert result["shape"] == [2, 2]
        assert result["mean_abs"] == 0.25
        assert result["max_abs"] == 0.4
        assert result["nan_count"] == 0
        assert result["inf_count"] == 0

    def test_nan_detection(self):
        result = summarize_action([[0.1, float("nan")], [0.3, 0.4]])
        assert result["nan_count"] == 1

    def test_inf_detection(self):
        result = summarize_action([[0.1, float("inf")], [0.3, 0.4]])
        assert result["inf_count"] == 1

    def test_empty_action(self):
        result = summarize_action([])
        assert result["shape"] == [0]
        assert result["mean_abs"] is None
        assert result["max_abs"] is None


class TestLiveMetricsCollector:
    def test_empty_summary(self):
        c = LiveMetricsCollector()
        s = c.summary()
        assert s["samples"] == 0
        assert s["mean_loop_ms"] is None
        assert s["p95_loop_ms"] is None
        assert s["processing_fps"] is None
        assert s["effective_fps"] is None

    def test_add_and_summary(self):
        c = LiveMetricsCollector()
        c.add(LiveMetricSample(step=0, ts="t0", loop_ms=10.0, runner_ms=5.0))
        c.add(LiveMetricSample(step=1, ts="t1", loop_ms=20.0, runner_ms=8.0))
        s = c.summary()
        assert s["samples"] == 2
        assert s["mean_loop_ms"] == 15.0
        assert s["max_loop_ms"] == 20.0
        assert s["mean_runner_ms"] == 6.5

    def test_percentile(self):
        c = LiveMetricsCollector()
        for i in range(20):
            c.add(LiveMetricSample(step=i, ts=f"t{i}", loop_ms=float(i + 1)))
        s = c.summary()
        assert s["p50_loop_ms"] is not None
        assert s["p95_loop_ms"] is not None
        assert s["p95_loop_ms"] >= s["p50_loop_ms"]

    def test_processing_fps(self):
        c = LiveMetricsCollector()
        # 10 samples at 100ms each = 1 second processing → 10 fps
        for i in range(10):
            c.add(LiveMetricSample(step=i, ts=f"t{i}", loop_ms=100.0))
        s = c.summary()
        assert s["processing_fps"] is not None
        assert abs(s["processing_fps"] - 10.0) < 1.0

    def test_effective_fps_from_wall_duration(self):
        """effective_fps uses wall-clock duration (includes sleep/pacing)."""
        c = LiveMetricsCollector()
        # 10 samples, processing time = 1s, but wall duration = 2s (sleep)
        for i in range(10):
            c.add(LiveMetricSample(step=i, ts=f"t{i}", loop_ms=100.0))
        s = c.summary(wall_duration_s=2.0)
        assert s["processing_fps"] is not None
        assert abs(s["processing_fps"] - 10.0) < 1.0
        assert s["effective_fps"] is not None
        assert abs(s["effective_fps"] - 5.0) < 1.0

    def test_effective_fps_none_without_wall_duration(self):
        """Without wall_duration_s, effective_fps is None (processing_fps only)."""
        c = LiveMetricsCollector()
        c.add(LiveMetricSample(step=0, ts="t0", loop_ms=100.0))
        s = c.summary()
        assert s["processing_fps"] is not None
        assert s["effective_fps"] is None

    def test_processing_fps_none_when_no_loop_ms(self):
        """When samples have no loop_ms, processing_fps is None."""
        c = LiveMetricsCollector()
        c.add(LiveMetricSample(step=0, ts="t0", loop_ms=None))
        s = c.summary(wall_duration_s=1.0)
        assert s["processing_fps"] is None
        assert s["effective_fps"] is not None  # still computable from wall_duration

    def test_nan_actions_accumulate(self):
        c = LiveMetricsCollector()
        c.add(LiveMetricSample(step=0, ts="t0", action_nan_count=1))
        c.add(LiveMetricSample(step=1, ts="t1", action_nan_count=2))
        s = c.summary()
        assert s["nan_actions"] == 3

    def test_inf_actions_accumulate(self):
        c = LiveMetricsCollector()
        c.add(LiveMetricSample(step=0, ts="t0", action_inf_count=1))
        s = c.summary()
        assert s["inf_actions"] == 1

    def test_shape_change_detection(self):
        c = LiveMetricsCollector()
        c.add(LiveMetricSample(step=0, ts="t0", action_shape=[16, 7]))
        c.add(LiveMetricSample(step=1, ts="t1", action_shape=[16, 7]))
        c.add(LiveMetricSample(step=2, ts="t2", action_shape=[8, 7]))  # change!
        s = c.summary()
        assert s["shape_changes"] == 1

    def test_latency_spikes(self):
        c = LiveMetricsCollector()
        # Most steps are ~10ms, one is 100ms (spike)
        for i in range(10):
            c.add(LiveMetricSample(step=i, ts=f"t{i}", loop_ms=10.0))
        c.add(LiveMetricSample(step=10, ts="t10", loop_ms=100.0))
        s = c.summary()
        assert s["latency_spikes"] >= 1
