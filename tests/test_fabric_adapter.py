# test_fabric_adapter.py — tests for the fabric adapter.

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from lerobot_coreai.fabric_adapter import FabricExportConfig, run_fabric_export
from lerobot_coreai.errors import CoreAIPolicyError


class TestFabricAdapter:
    def test_missing_cli_and_module_gives_install_hint(self):
        """When neither CLI nor Python module is available, error should mention install."""
        with patch("lerobot_coreai.fabric_adapter.shutil.which", return_value=None), \
             patch("builtins.__import__", side_effect=ImportError):
            config = FabricExportConfig(
                torch_policy_path="test",
                output_dir=Path("/tmp/test"),
            )
            with pytest.raises(CoreAIPolicyError, match="install"):
                run_fabric_export(config)

    def test_missing_cli_clear_error(self):
        """Without CLI or module, error should mention --skip-fabric."""
        # Mock: no CLI binary, import fails
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "coreai_fabric" or name.startswith("coreai_fabric."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("lerobot_coreai.fabric_adapter.shutil.which", return_value=None), \
             patch("builtins.__import__", side_effect=mock_import):
            config = FabricExportConfig(torch_policy_path="test", output_dir=Path("/tmp/test"))
            with pytest.raises(CoreAIPolicyError, match="--skip-fabric"):
                run_fabric_export(config)
