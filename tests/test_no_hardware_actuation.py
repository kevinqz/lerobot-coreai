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

# Stronger patterns that flag actuation egress in executable source (spec §16).
# These are forbidden even as substrings, with context to avoid false positives.
FORBIDDEN_SOURCE_PATTERNS = [
    ".send_action(",
    "serial.Serial",
    "dynamixel_sdk",
    "motor_bus",
    "teleop",
    "write_position",
    "write_goal_position",
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

    def test_no_forbidden_source_patterns(self):
        """Check for actuation egress patterns in executable source (spec §16).

        These patterns flag direct hardware command egress. 'send_action' may
        appear only in docs/comments/tests explaining it is forbidden, not in
        executable source. The '.send_action(' form catches method calls.
        """
        src_dir = Path(__file__).parent.parent / "src" / "lerobot_coreai"
        violations = []
        for py in src_dir.rglob("*.py"):
            content = py.read_text()
            for pattern in FORBIDDEN_SOURCE_PATTERNS:
                if pattern.lower() in content.lower():
                    violations.append(f"{py.name}: found '{pattern}'")
        assert not violations, f"Forbidden actuation patterns found:\n" + "\n".join(violations)
