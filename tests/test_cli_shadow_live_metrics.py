# test_cli_shadow_live_metrics.py — tests for v0.7.2 CLI args.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli
from lerobot_coreai.shadow import ShadowResult
from lerobot_coreai.shadow_quality import ShadowQualityConfig


def _mock_shadow_result(tmp_path):
    r = MagicMock(spec=ShadowResult)
    r.ok = True
    r.report = {
        "ok": True,
        "schema_version": "lerobot-coreai.shadow_report.v0",
        "metrics": {"observations_read": 3, "actions_generated": 3, "actions_blocked": 3, "actions_sent": 0},
        "live_metrics": {"samples": 3, "mean_loop_ms": 10.0},
        "quality": {"passed": True, "checks": []},
    }
    r.report_path = tmp_path / "shadow_report.json"
    r.trace_path = tmp_path / "shadow_trace.jsonl"
    r.actions_path = tmp_path / "actions.jsonl"
    r.observations_path = tmp_path / "observations.jsonl"
    r.blocked_actions_path = tmp_path / "blocked_actions.jsonl"
    return r


class TestCLIShadowLiveMetrics:
    def test_live_flag_parsed(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--live",
            ])
        config = mock_run.call_args[0][0]
        assert config.live is True

    def test_live_every_parsed(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--live",
                "--live-every", "5",
            ])
        config = mock_run.call_args[0][0]
        assert config.live_every == 5

    def test_quality_args_build_config(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--quality.max-runner-p95-ms", "50",
                "--quality.min-effective-fps", "5",
            ])
        config = mock_run.call_args[0][0]
        assert config.quality_config is not None
        assert config.quality_config.max_runner_p95_ms == 50.0
        assert config.quality_config.min_effective_fps == 5.0

    def test_quality_fail_on_quality(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--quality.max-runner-p95-ms", "50",
                "--quality.fail-on-quality",
            ])
        config = mock_run.call_args[0][0]
        assert config.fail_on_quality is True

    def test_no_quality_args_means_no_config(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
            ])
        config = mock_run.call_args[0][0]
        assert config.quality_config is None

    def test_adapter_required_keys_parsed(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--adapter.required-keys", "observation.images.wrist,observation.state",
            ])
        config = mock_run.call_args[0][0]
        assert config.required_keys == ["observation.images.wrist", "observation.state"]

    def test_adapter_image_map_parsed(self, tmp_path):
        mock_result = _mock_shadow_result(tmp_path)
        with patch("lerobot_coreai.cli.run_shadow_mode", return_value=mock_result) as mock_run:
            cli.main([
                "shadow",
                "--policy.path", "test",
                "--observation-source", "folder",
                "--frames-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "run"),
                "--adapter.image-map", "camera_front=observation.images.front",
            ])
        config = mock_run.call_args[0][0]
        assert config.adapter_image_keys == {"camera_front": "observation.images.front"}
