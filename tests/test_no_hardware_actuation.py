# test_no_hardware_actuation.py — negative test: no hardware/actuation tokens in src/.

import pytest
from pathlib import Path

BANNED = [
    "dynamixel",
    "serial",
    "send_action",
    "robot.connect",
    "teleop",
    "motor_bus",
    "motorbridge",
    "feetech",
]


class TestNoHardwareActuation:
    def test_no_banned_tokens_in_src(self):
        """Search all .py files under src/lerobot_coreai/ for banned hardware tokens."""
        src_dir = Path(__file__).parent.parent / "src" / "lerobot_coreai"
        violations = []
        for py in src_dir.rglob("*.py"):
            content = py.read_text().lower()
            for token in BANNED:
                if token.lower() in content:
                    violations.append(f"{py.name}: found '{token}'")
        assert not violations, f"Banned hardware tokens found:\n" + "\n".join(violations)
