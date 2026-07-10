# test_config_and_modes.py — config invariants + runtime binding modes (v1.3.6).

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")
import torch  # noqa: E402

from lerobot_policy_coreai_bridge import CoreAIBridgeConfig, CoreAIBridgePolicy  # noqa: E402
from lerobot_policy_coreai_bridge.modeling_coreai_bridge import PluginBindingError  # noqa: E402


# --- config invariants ---

def test_invalid_binding_mode_fails():
    with pytest.raises(ValueError):
        CoreAIBridgeConfig(runtime_binding_mode="bogus")


def test_contradictory_horizons_fail():
    with pytest.raises(ValueError):
        CoreAIBridgeConfig(action_horizon=5, expected_action_horizon=3)


def test_effective_horizon_prefers_expected():
    cfg = CoreAIBridgeConfig(expected_action_horizon=3)
    assert cfg.effective_action_horizon() == 3
    assert CoreAIBridgeConfig().effective_action_horizon() is None


def test_valid_modes_accepted():
    for mode in ("strict", "legacy", "in_memory"):
        assert CoreAIBridgeConfig(runtime_binding_mode=mode).runtime_binding_mode == mode


# --- binding modes at inference ---

class _FakeNoRunner:
    """In-process CoreAI policy with NO RunnerClient (runner attr absent)."""
    def __init__(self):
        self.robot_type = "so100"
        self.manifest = {"contracts": {"action": {
            "representation": "chunk", "horizon": 3, "action_dim": 7}}}

    def predict_action_chunk(self, batch, runner_options=None, **kw):
        return [[float(i)] * 7 for i in range(3)]

    def reset(self):
        pass


def _obs():
    return {"observation.state": torch.zeros(1, 7)}


def test_strict_mode_without_runner_fails():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="strict"),
                           coreai_policy=_FakeNoRunner())
    with pytest.raises(PluginBindingError):
        p.select_action(_obs())


def test_in_memory_mode_without_runner_works():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"),
                           coreai_policy=_FakeNoRunner())
    assert p.select_action(_obs()).shape == (1, 7)
