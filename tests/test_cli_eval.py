# test_cli_eval.py — tests for lerobot-coreai eval CLI command.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli


class TestCLIEval:
    def test_eval_success(self, tmp_path, capsys, valid_manifest_dict):
        from lerobot_coreai.eval import EvalResult

        mock_result = MagicMock(spec=EvalResult)
        mock_result.ok = True
        mock_result.report = {"ok": True, "metrics": {"frames_processed": 5, "actions_generated": 5, "actions_failed": 0}}
        mock_result.actions_path = tmp_path / "actions.jsonl"
        mock_result.trace_path = tmp_path / "trace.jsonl"
        mock_result.report_path = tmp_path / "report.json"

        with patch("lerobot_coreai.cli.run_lerobot_dataset_eval", return_value=mock_result):
            rc = cli.main([
                "eval",
                "--policy.path", "kevinqz/EVO1-SO100-CoreAI",
                "--dataset.repo_id", "lerobot/test",
                "--runner.url", "http://localhost:8710",
                "--max-frames", "5",
                "--output-dir", str(tmp_path / "eval"),
            ])
            assert rc == 0
            out = capsys.readouterr().out
            assert "No robot commands sent" in out
            assert "Eval completed" in out

    def test_eval_json_output(self, tmp_path, capsys):
        from lerobot_coreai.eval import EvalResult

        mock_result = MagicMock(spec=EvalResult)
        mock_result.ok = True
        mock_result.report = {"ok": True, "schema_version": "lerobot-coreai.eval_report.v0", "metrics": {}}
        mock_result.actions_path = tmp_path / "a.jsonl"
        mock_result.trace_path = tmp_path / "t.jsonl"
        mock_result.report_path = tmp_path / "r.json"

        with patch("lerobot_coreai.cli.run_lerobot_dataset_eval", return_value=mock_result):
            rc = cli.main([
                "eval",
                "--policy.path", "test",
                "--dataset.repo_id", "test",
                "--json",
            ])
            assert rc == 0
            out = capsys.readouterr().out
            data = json.loads(out)
            assert data["ok"] is True

    def test_eval_failure_prints_no_robot(self, tmp_path, capsys):
        from lerobot_coreai.errors import CoreAIPolicyError

        with patch("lerobot_coreai.cli.run_lerobot_dataset_eval",
                    side_effect=CoreAIPolicyError("dataset missing")):
            rc = cli.main([
                "eval",
                "--policy.path", "test",
                "--dataset.repo_id", "bad/dataset",
            ])
            assert rc == 1
            err = capsys.readouterr().err
            assert "No robot commands were sent" in err
