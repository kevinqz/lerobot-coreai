# test_discovery.py — real out-of-tree plugin discovery (v1.3.2).
#
# LeRobot discovers installed lerobot_policy_* distributions and imports them so
# they self-register. This proves that path in a CLEAN subprocess that does NOT
# manually import the companion before discovery.

import subprocess
import sys

import pytest

pytest.importorskip("lerobot")

_SCRIPT = """
import sys
# Must NOT import lerobot_policy_coreai_bridge manually before discovery.
assert "lerobot_policy_coreai_bridge" not in sys.modules
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.configs.policies import PreTrainedConfig
register_third_party_plugins()
choices = PreTrainedConfig.get_known_choices()
assert "coreai_bridge" in choices, f"coreai_bridge not discovered: {sorted(choices)}"
# The reserved bare name must NOT be registered.
assert "coreai" not in choices
print("DISCOVERY_OK")
"""


def _lerobot_has_plugin_discovery():
    try:
        from lerobot.utils.import_utils import register_third_party_plugins  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _lerobot_has_plugin_discovery(),
                    reason="register_third_party_plugins not available")
def test_discovery_in_clean_subprocess():
    proc = subprocess.run([sys.executable, "-c", _SCRIPT],
                          capture_output=True, text=True, timeout=180)
    assert "DISCOVERY_OK" in proc.stdout, (
        f"discovery failed.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-2000:]}")
