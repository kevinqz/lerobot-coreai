# test_safety.py — tests for safety mode enforcement.

import pytest

from lerobot_coreai.safety import ensure_mode_supported_for_v03, assert_no_physical_actuation_available
from lerobot_coreai.errors import SafetyError


class TestSafetyModes:
    def test_dry_run_allowed(self):
        ensure_mode_supported_for_v03("dry_run")

    def test_shadow_blocked(self):
        with pytest.raises(SafetyError, match="No robot commands were sent"):
            ensure_mode_supported_for_v03("shadow")

    def test_sim_blocked(self):
        with pytest.raises(SafetyError, match="No robot commands were sent"):
            ensure_mode_supported_for_v03("sim")

    def test_real_blocked(self):
        with pytest.raises(SafetyError, match="not implemented"):
            ensure_mode_supported_for_v03("real")

    def test_real_blocked_even_with_confirmation(self):
        with pytest.raises(SafetyError, match="No robot commands were sent"):
            ensure_mode_supported_for_v03("real", confirm_real_robot_actuation=True)

    def test_assert_no_physical_actuation(self):
        assert assert_no_physical_actuation_available() is None
