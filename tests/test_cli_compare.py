# test_cli_compare.py — tests for lerobot-coreai compare CLI.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli


class TestCLICompare:
    def test_compare_success(self, tmp_path, capsys):
        from lerobot_coreai.compare import CompareResult

        mock_result = MagicMock(spec=CompareResult)
        mock_result.ok = True
        mock_result.report = {
            "ok": True, "metrics": {"frames_compared": 5, "frames_passed": 5, "frames_failed": 0,
                                    "min_cosine_similarity": 0.9999},
            "claims": {"proves_numeric_action_fidelity": True}
        }
        mock_result.actions_path = tmp_path / "actions.jsonl"
        mock_result.trace_path = tmp_path / "trace.jsonl"
        mock_result.report_path = tmp_path / "report.json"

        with patch("lerobot_coreai.cli.run_lerobot_policy_compare", return_value=mock_result):
            rc = cli.main([
                "compare",
                "--torch.policy.path", "lerobot/test",
                "--coreai.policy.path", "kevinqz/test-CoreAI",
                "--dataset.repo_id", "lerobot/dataset",
                "--runner.url", "http://localhost:8710",
                "--max-frames", "5",
                "--output-dir", str(tmp_path / "cmp"),
            ])
            assert rc == 0
            out = capsys.readouterr().out
            assert "Compare completed" in out
            assert "No robot commands sent" in out

    def test_compare_json_output(self, tmp_path, capsys):
        from lerobot_coreai.compare import CompareResult

        mock_result = MagicMock(spec=CompareResult)
        mock_result.ok = True
        mock_result.report = {"ok": True, "schema_version": "lerobot-coreai.compare_report.v0"}

        with patch("lerobot_coreai.cli.run_lerobot_policy_compare", return_value=mock_result):
            rc = cli.main([
                "compare",
                "--torch.policy.path", "t",
                "--coreai.policy.path", "c",
                "--dataset.repo_id", "d",
                "--json",
            ])
            assert rc == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["ok"] is True

    def test_compare_failure_prints_no_robot(self, tmp_path, capsys):
        from lerobot_coreai.errors import CoreAIPolicyError

        with patch("lerobot_coreai.cli.run_lerobot_policy_compare",
                    side_effect=CoreAIPolicyError("policy missing")):
            rc = cli.main([
                "compare",
                "--torch.policy.path", "t",
                "--coreai.policy.path", "c",
                "--dataset.repo_id", "d",
            ])
            assert rc == 1
            err = capsys.readouterr().err
            assert "No robot commands were sent" in err
