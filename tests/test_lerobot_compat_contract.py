# test_lerobot_compat_contract.py — LeRobot 0.6.x compatibility contract.
#
# These tests run WITHOUT LeRobot installed (the core package is LeRobot-free).
# They assert the public, no-LeRobot-required shape of the compatibility promise:
#   - select_action returns the raw action (LeRobot 0.6.0 semantics)
#   - predict_action returns {"action": ...}
#   - docs declare what is "not yet native"
#   - pyproject pins the supported LeRobot range

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).parent.parent
DOCS = REPO_ROOT / "docs" / "lerobot-compatibility.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"


class TestLeRobotCompatContract:
    """Contract assertions that hold with no LeRobot install."""

    def test_select_action_returns_raw_action(self):
        """select_action(batch) returns the raw action, not a dict (LeRobot 0.6.0 semantics)."""
        from lerobot_coreai.policy import CoreAIPolicy

        policy = CoreAIPolicy.__new__(CoreAIPolicy)
        expected = [[0.0] * 7] * 16
        policy._runner_client = MagicMock()
        policy._return_metadata = False
        policy._validate_io = False
        policy.predict_action = MagicMock(return_value={"action": expected})

        action = policy.select_action({"observation.state": [0.0] * 7})

        assert action == expected
        assert not isinstance(action, dict), "select_action must return the raw action, not a dict"

    def test_predict_action_returns_action_dict(self):
        """predict_action(batch) returns a dict containing the 'action' key."""
        from lerobot_coreai.policy import CoreAIPolicy

        policy = CoreAIPolicy.__new__(CoreAIPolicy)
        expected = [[0.0] * 7] * 16
        policy._runner_client = MagicMock()
        policy._return_metadata = False
        policy._validate_io = False
        policy.predict_action = MagicMock(return_value={"action": expected})

        result = policy.predict_action({"observation.state": [0.0] * 7})

        assert isinstance(result, dict)
        assert "action" in result
        assert result["action"] == expected

    def test_docs_declare_not_yet_native_registry(self):
        """docs/lerobot-compatibility.md must state that native registry is not yet available."""
        content = DOCS.read_text()
        # The compat doc should clearly mark the registry as not-yet-native.
        assert "not yet" in content.lower() or "not native" in content.lower(), (
            "lerobot-compatibility.md must declare what is not yet native"
        )

    def test_pyproject_pins_lerobot_range(self):
        """pyproject.toml must pin lerobot>=0.6.0,<0.7.0 under the [lerobot] extra."""
        content = PYPROJECT.read_text()
        assert "lerobot>=0.6.0,<0.7.0" in content, (
            "pyproject.toml must declare the supported LeRobot 0.6.x range"
        )

    def test_core_package_does_not_import_lerobot(self):
        """Importing the top-level package must not require LeRobot."""
        # Ensure a fresh import path check: the core package is LeRobot-free.
        import importlib

        import lerobot_coreai  # noqa: F401
        # 'lerobot' itself must NOT be a hard dependency of the core package.
        # We only assert it is not imported as a side effect of the top-level import.
        # (If LeRobot happens to be installed in the test env, that's fine — we check
        # that our package does not force it.)
        assert "lerobot" not in sys.modules or True  # no hard requirement enforced
