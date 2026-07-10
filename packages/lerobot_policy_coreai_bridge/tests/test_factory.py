# test_factory.py — official factory / from_pretrained runtime binding (v1.3.1).
# Runs where lerobot + torch are installed (stable CI); skips otherwise.

from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_policy_config  # noqa: E402

from lerobot_policy_coreai_bridge import CoreAIBridgeConfig, CoreAIBridgePolicy  # noqa: E402
from lerobot_policy_coreai_bridge.modeling_coreai_bridge import PluginBindingError  # noqa: E402


class _FakeCoreAI:
    def __init__(self, horizon=3, action_dim=7):
        self._chunk = [[float(i)] * action_dim for i in range(horizon)]
        self.robot_type = "so100"
        self.manifest = {"contracts": {"action": {
            "representation": "chunk", "horizon": horizon, "action_dim": action_dim}}}

    def predict_action_chunk(self, batch, runner_options=None, **kw):
        return self._chunk

    def reset(self):
        pass


# --- Official factory resolution (no monkeypatch) ---

def test_get_choice_class_resolves():
    assert PreTrainedConfig.get_choice_class("coreai_bridge") is CoreAIBridgeConfig


def test_make_policy_config_resolves():
    cfg = make_policy_config("coreai_bridge")
    assert isinstance(cfg, CoreAIBridgeConfig)


def test_get_policy_class_resolves():
    assert get_policy_class("coreai_bridge") is CoreAIBridgePolicy


# --- Constructor accepts LeRobot make_policy kwargs ---

def test_ctor_accepts_dataset_kwargs():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=_FakeCoreAI(),
                           dataset_stats={"x": 1}, dataset_meta=object())
    assert p.dataset_stats == {"x": 1}


# --- from_pretrained runtime binding ---

def test_from_pretrained_binds_runner(monkeypatch):
    monkeypatch.setenv("COREAI_RUNNER_URL", "http://127.0.0.1:8710")
    cfg = CoreAIBridgeConfig(coreai_artifact="kevinqz/EVO1-SO100-CoreAI",
                             expected_action_dim=7, expected_action_horizon=3,
                             expected_robot_type="so100",
                             runtime_binding_mode="in_memory")
    with patch("lerobot_coreai.policy.CoreAIPolicy.from_pretrained",
               return_value=_FakeCoreAI()):
        policy = CoreAIBridgePolicy.from_pretrained("kevinqz/EVO1-SO100-CoreAI", config=cfg)
    assert policy.coreai_policy is not None
    a = policy.select_action({"observation.state": torch.zeros(1, 7)})
    assert a.shape == (1, 7)


def test_from_pretrained_fails_without_runner_env(monkeypatch):
    monkeypatch.delenv("COREAI_RUNNER_URL", raising=False)
    cfg = CoreAIBridgeConfig(coreai_artifact="x")
    with pytest.raises(PluginBindingError):
        CoreAIBridgePolicy.from_pretrained("x", config=cfg)


def test_cross_binding_mismatch_fails(monkeypatch):
    monkeypatch.setenv("COREAI_RUNNER_URL", "http://127.0.0.1:8710")
    cfg = CoreAIBridgeConfig(coreai_artifact="x", expected_action_dim=99)  # wrong
    with patch("lerobot_coreai.policy.CoreAIPolicy.from_pretrained",
               return_value=_FakeCoreAI()):
        with pytest.raises(PluginBindingError):
            CoreAIBridgePolicy.from_pretrained("x", config=cfg)


# --- Batch guard + device ---

def test_batch_size_gt_1_fails_clearly():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(batch_mode="single_only"),
                           coreai_policy=_FakeCoreAI())
    with pytest.raises(PluginBindingError):
        p.select_action({"observation.state": torch.zeros(4, 7)})  # B=4


def test_action_tensor_on_policy_device():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=_FakeCoreAI())
    a = p.select_action({"observation.state": torch.zeros(1, 7)})
    assert a.device == p._sentinel.device
    chunk = p.predict_action_chunk({"observation.state": torch.zeros(1, 7)})
    assert chunk.ndim == 3  # (B, H, A)
