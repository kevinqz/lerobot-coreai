# test_lerobot060_smoke.py — LeRobot 0.6.x integration smoke tests.
#
# Gated behind LEROBOT_INTEGRATION=1 so they never run in normal CI.
# Run with:
#   LEROBOT_INTEGRATION=1 pytest tests/integration/test_lerobot060_smoke.py

import os
import sys

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LEROBOT_INTEGRATION") != "1",
    reason="Set LEROBOT_INTEGRATION=1 to run LeRobot integration smoke tests.",
)


class TestLeRobot060Smoke:
    """Smoke checks that run only when LeRobot is installed and opted-in."""

    def test_python_is_312_plus(self):
        """LeRobot 0.6.0 requires Python 3.12+."""
        assert sys.version_info >= (3, 12), "LeRobot 0.6.x requires Python >= 3.12"

    def test_lerobot_imports(self):
        """lerobot must be importable."""
        import lerobot  # noqa: F401

    def test_lerobot_dataset_constructor_available(self):
        """The public LeRobotDataset constructor must be importable from the canonical path."""
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

    def test_lerobot_version_in_supported_range(self):
        """Installed LeRobot must be in the 0.6.x range."""
        import lerobot

        version = getattr(lerobot, "__version__", None)
        assert version is not None, "lerobot.__version__ is not exposed"
        major, minor = version.split(".")[:2]
        assert (int(major), int(minor)) == (0, 6), (
            f"Expected LeRobot 0.6.x, got {version}"
        )
