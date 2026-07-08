# test_cli_export.py — tests for lerobot-coreai export CLI.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai import cli


class TestCLIExport:
    def test_export_success(self, tmp_path, capsys):
        from lerobot_coreai.export import ExportResult

        mock_result = MagicMock(spec=ExportResult)
        mock_result.ok = True
        mock_result.report = {"ok": True, "verification": {"manifest_valid": True}}
        mock_result.report_path = tmp_path / "report.json"
        mock_result.trace_path = tmp_path / "trace.jsonl"
        mock_result.manifest_path = tmp_path / "manifest.json"
        mock_result.artifact_path = tmp_path / "model.aimodel"

        artifact = tmp_path / "model.aimodel"
        artifact.mkdir()

        with patch("lerobot_coreai.cli.run_coreai_export_pipeline", return_value=mock_result):
            rc = cli.main([
                "export",
                "--torch.policy.path", "lerobot/test",
                "--skip-fabric",
                "--existing-artifact", str(artifact),
                "--output-dir", str(tmp_path / "export"),
            ])
            assert rc == 0
            out = capsys.readouterr().out
            assert "No robot commands sent" in out

    def test_export_json_output(self, tmp_path, capsys):
        from lerobot_coreai.export import ExportResult

        mock_result = MagicMock(spec=ExportResult)
        mock_result.ok = True
        mock_result.report = {"ok": True, "schema_version": "lerobot-coreai.export_report.v0"}

        with patch("lerobot_coreai.cli.run_coreai_export_pipeline", return_value=mock_result):
            rc = cli.main([
                "export",
                "--torch.policy.path", "test",
                "--output-dir", str(tmp_path / "e"),
                "--json",
            ])
            assert rc == 0
            data = json.loads(capsys.readouterr().out)
            assert data["ok"] is True

    def test_export_failure_prints_no_robot(self, tmp_path, capsys):
        from lerobot_coreai.errors import CoreAIPolicyError

        with patch("lerobot_coreai.cli.run_coreai_export_pipeline",
                    side_effect=CoreAIPolicyError("fabric missing")):
            rc = cli.main([
                "export",
                "--torch.policy.path", "test",
                "--output-dir", str(tmp_path / "e"),
            ])
            assert rc == 1
            err = capsys.readouterr().err
            assert "No robot commands were sent" in err
