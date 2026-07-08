# test_no_hardware_actuation.py — negative test: no hardware/actuation tokens in src/.

import pytest
from pathlib import Path

BANNED_IMPORTS = [
    "import serial",
    "from serial",
    "import dynamixel",
    "from dynamixel",
    "import feetech",
    "from feetech",
]

BANNED_TOKENS = [
    "dynamixel",
    "send_action",
    "robot.connect",
    "teleop",
    "motor_bus",
    "motorbridge",
    "feetech",
    "pyserial",
]


class TestNoHardwareActuation:
    def test_no_banned_tokens_in_src(self):
        """Search all .py files under src/lerobot_coreai/ for banned hardware tokens.

        Checks for both import statements and function/class references.
        'serial' as a substring is allowed (e.g. 'serialization', 'JSON serializable')
        but 'import serial' or 'from serial' is not.
        """
        src_dir = Path(__file__).parent.parent / "src" / "lerobot_coreai"
        violations = []
        for py in src_dir.rglob("*.py"):
            content = py.read_text().lower()
            for token in BANNED_IMPORTS:
                if token.lower() in content:
                    violations.append(f"{py.name}: found '{token}'")
            for token in BANNED_TOKENS:
                if token.lower() in content:
                    violations.append(f"{py.name}: found '{token}'")
        assert not violations, f"Banned hardware tokens found:\n" + "\n".join(violations)
