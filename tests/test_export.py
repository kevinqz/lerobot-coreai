# test_export.py — tests for export pipeline with mocks.

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.export import ExportConfig, run_coreai_export_pipeline
from lerobot_coreai.errors import CoreAIPolicyError


class TestExportMinimal:
    def test_skip_fabric_with_existing_artifact(self, tmp_path):
        """Minimal export: skip fabric, use existing artifact."""
        # Create a fake artifact.
        artifact = tmp_path / "model.aimodel"
        artifact.mkdir()
        (artifact / "metadata.json").write_text("{}")

        config = ExportConfig(
            torch_policy_path="lerobot/test",
            output_dir=tmp_path / "export",
            skip_fabric=True,
            existing_artifact=artifact,
            overwrite=True,
        )
        result = run_coreai_export_pipeline(config)

        assert result.ok is True
        assert result.report_path.exists()
        assert result.trace_path.exists()
        assert result.report["safety"]["actions_sent"] == 0

    def test_skip_fabric_without_artifact_fails(self, tmp_path):
        config = ExportConfig(
            torch_policy_path="lerobot/test",
            output_dir=tmp_path / "export",
            skip_fabric=True,
            overwrite=True,
        )
        with pytest.raises(CoreAIPolicyError, match="existing-artifact"):
            run_coreai_export_pipeline(config)

    def test_export_writes_manifest(self, tmp_path):
        artifact = tmp_path / "model.aimodel"
        artifact.mkdir()
        (artifact / "metadata.json").write_text("{}")

        config = ExportConfig(
            torch_policy_path="lerobot/test",
            output_dir=tmp_path / "export",
            skip_fabric=True,
            existing_artifact=artifact,
            policy_type="act",
            robot_type="so100",
            overwrite=True,
        )
        result = run_coreai_export_pipeline(config)

        assert result.manifest_path is not None
        assert result.manifest_path.exists()
        manifest = json.loads(result.manifest_path.read_text())
        assert manifest["policy"]["type"] == "act"
        assert manifest["robot"]["type"] == "so100"

    def test_export_safety_invariants(self, tmp_path):
        artifact = tmp_path / "model.aimodel"
        artifact.mkdir()

        config = ExportConfig(
            torch_policy_path="test", output_dir=tmp_path / "e",
            skip_fabric=True, existing_artifact=artifact,
            overwrite=True,
        )
        result = run_coreai_export_pipeline(config)

        assert result.report["safety"]["actions_sent"] == 0
        assert result.report["safety"]["physical_actuation_possible"] is False
        assert result.report["claims"]["proves_task_success"] is False
        assert result.report["claims"]["proves_robot_safety"] is False


class TestExportPublishReady:
    def test_publish_ready_creates_folder(self, tmp_path):
        artifact = tmp_path / "model.aimodel"
        artifact.mkdir()
        (artifact / "metadata.json").write_text("{}")

        config = ExportConfig(
            torch_policy_path="lerobot/test",
            output_dir=tmp_path / "export",
            skip_fabric=True,
            existing_artifact=artifact,
            policy_type="act",
            robot_type="so100",
            model_id="test-model",
            publish_ready=True,
            overwrite=True,
        )
        result = run_coreai_export_pipeline(config)

        publish_dir = tmp_path / "export" / "publish"
        assert publish_dir.exists()
        assert (publish_dir / "lerobot-coreai.json").exists()
        assert (publish_dir / "README.md").exists()
        assert result.report["claims"]["publish_ready"] is True
