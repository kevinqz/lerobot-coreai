# test_lerobot060_smoke.py — LeRobot 0.6.x integration smoke tests.
#
# These run automatically when LeRobot is installed (the recommended test
# environment — install with `pip install -e ".[lerobot,test]"`). They skip
# themselves cleanly when LeRobot is absent, so a minimal core-only CI run
# (no LeRobot) stays green.
#
# To force-skip even when LeRobot is installed, set LEROBOT_INTEGRATION_SKIP=1.

import os
import sys

import pytest

# Skip the whole module if LeRobot is not importable, or if explicitly skipped.
if os.environ.get("LEROBOT_INTEGRATION_SKIP") == "1":
    pytestmark = pytest.mark.skip(reason="LEROBOT_INTEGRATION_SKIP=1 set")
else:
    pytestmark = pytest.mark.skipif(
        sys.version_info < (3, 12),
        reason="LeRobot 0.6.x requires Python >= 3.12",
    )


class TestLeRobot060Smoke:
    """Smoke checks that run automatically when LeRobot is installed.

    Each test uses pytest.importorskip so it skips cleanly (not fails) when
    LeRobot is genuinely absent — keeping a minimal core-only CI green.
    """

    def test_python_is_312_plus(self):
        """LeRobot 0.6.0 requires Python 3.12+."""
        assert sys.version_info >= (3, 12), "LeRobot 0.6.x requires Python >= 3.12"

    def test_lerobot_imports(self):
        """lerobot must be importable."""
        pytest.importorskip("lerobot")

    def test_lerobot_dataset_constructor_available(self):
        """The public LeRobotDataset constructor must be importable from the canonical path."""
        pytest.importorskip("lerobot")
        pytest.importorskip("lerobot.datasets.lerobot_dataset")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

    def test_lerobot_version_in_supported_range(self):
        """Installed LeRobot must be in the 0.6.x range."""
        lerobot = pytest.importorskip("lerobot")
        version = getattr(lerobot, "__version__", None)
        assert version is not None, "lerobot.__version__ is not exposed"
        major, minor = version.split(".")[:2]
        assert (int(major), int(minor)) == (0, 6), (
            f"Expected LeRobot 0.6.x, got {version}"
        )
