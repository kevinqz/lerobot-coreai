# test_no_hardware_actuation.py — negative test: no UNGATED hardware/actuation egress.
#
# Through v0.9.x there was zero actuation code anywhere. v1.0.0 introduces the
# FIRST sanctioned egress path — guarded real mode — which is confined to a small
# allowlist of modules (real_egress / robot_adapters / real_mode / real_preflight
# / real_reports). This test enforces two things:
#   1. hardware-driver imports (serial/dynamixel/feetech/…) are banned EVERYWHERE
#      — the project never talks to motors directly, even in real mode.
#   2. action-egress tokens (send_action, robot.connect, teleop, motor_bus, …)
#      appear ONLY inside the guarded real-mode allowlist, never elsewhere.

from pathlib import Path

# Files where guarded real egress is intentional and gated (v1.0.0).
REAL_MODE_ALLOWLIST = {
    "real_egress.py", "robot_adapters.py", "real_mode.py",
    "real_preflight.py", "real_reports.py",
}

# Hardware-driver imports: banned in EVERY source file, including real mode.
# The project reaches hardware only through an operator-provided external adapter,
# never a bundled motor/serial driver.
BANNED_EVERYWHERE = [
    "import serial", "from serial", "import dynamixel", "from dynamixel",
    "import feetech", "from feetech", "dynamixel", "feetech", "pyserial",
    "dynamixel_sdk", "serial.serial", "motor_bus", "motorbridge",
    "write_position", "write_goal_position",
]

# Actuation-egress tokens: allowed ONLY inside the real-mode allowlist.
GATED_TO_REAL_MODE = ["send_action", "robot.connect", "teleop"]


def _iter_src():
    src_dir = Path(__file__).parent.parent / "src" / "lerobot_coreai"
    return src_dir.rglob("*.py")


class TestNoHardwareActuation:
    def test_no_hardware_driver_tokens_anywhere(self):
        violations = []
        for py in _iter_src():
            content = py.read_text().lower()
            for token in BANNED_EVERYWHERE:
                if token in content:
                    violations.append(f"{py.name}: found '{token}'")
        assert not violations, "Banned hardware-driver tokens found:\n" + "\n".join(violations)

    def test_actuation_egress_confined_to_real_mode(self):
        violations = []
        for py in _iter_src():
            if py.name in REAL_MODE_ALLOWLIST:
                continue  # guarded real mode is the sanctioned egress path
            content = py.read_text().lower()
            for token in GATED_TO_REAL_MODE:
                if token in content:
                    violations.append(f"{py.name}: found '{token}' outside real-mode allowlist")
        assert not violations, (
            "Actuation-egress tokens found OUTSIDE the guarded real-mode allowlist:\n"
            + "\n".join(violations))

    def test_real_egress_only_reachable_through_guard(self):
        # The only .send_action( call sites are the adapter definitions and the
        # RealEgressGuard. No other module may call an adapter's send_action.
        for py in _iter_src():
            if py.name in REAL_MODE_ALLOWLIST:
                continue
            assert ".send_action(" not in py.read_text(), (
                f"{py.name} calls .send_action() outside the real-mode allowlist")
