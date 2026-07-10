# test_plugin.py — official out-of-tree LeRobot plugin (v1.3.0).
# Runs where lerobot + torch are installed (the stable CI job); skips otherwise.

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402

from lerobot_policy_coreai_bridge import (  # noqa: E402
    CoreAIBridgeConfig, CoreAIBridgePolicy, make_coreai_bridge_pre_post_processors,
)


class _FakeCoreAI:
    """A stand-in CoreAI policy returning a fixed chunk [H, A]."""

    def __init__(self, horizon=3, action_dim=7):
        self._chunk = [[float(i)] * action_dim for i in range(horizon)]
        self.resets = 0

    def predict_action_chunk(self, batch, runner_options=None, **kw):
        return self._chunk

    def reset(self):
        self.resets += 1


def _policy():
    return CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=_FakeCoreAI())


def test_config_registered_under_coreai_bridge():
    assert "coreai_bridge" in PreTrainedConfig.get_known_choices()
    # The reserved bare "coreai" name must NOT be registered by this plugin.
    assert "coreai" not in PreTrainedConfig.get_known_choices()


def test_policy_is_pretrained_and_nn_module():
    p = _policy()
    assert isinstance(p, PreTrainedPolicy)
    assert isinstance(p, nn.Module)
    assert p.name == "coreai_bridge"
    assert p.config_class is CoreAIBridgeConfig


def test_select_action_returns_batched_tensor():
    p = _policy()
    a = p.select_action({"observation.state": [0.0] * 7})
    assert isinstance(a, torch.Tensor)
    assert a.ndim == 2 and a.shape[0] == 1 and a.shape[1] == 7  # (B, action_dim)


def test_select_action_drains_queue_then_refills():
    p = _policy()
    batch = {"observation.state": [0.0] * 7}
    first = [p.select_action(batch)[0, 0].item() for _ in range(3)]
    assert first == [0.0, 1.0, 2.0]  # per-timestep, in order
    # Queue exhausted → refills from a fresh chunk.
    assert p.select_action(batch)[0, 0].item() == 0.0


def test_predict_action_chunk_returns_tensor():
    p = _policy()
    chunk = p.predict_action_chunk({"observation.state": [0.0] * 7})
    assert isinstance(chunk, torch.Tensor)
    assert chunk.shape == (1, 3, 7)  # (B, H, A) — v1.3.1 tensor contract


def test_train_true_raises_but_eval_works():
    p = _policy()
    with pytest.raises(RuntimeError):
        p.train(True)
    assert p.eval() is p           # eval() -> train(False) must work
    assert p.train(False) is p


def test_forward_and_optim_raise():
    p = _policy()
    with pytest.raises(RuntimeError):
        p.forward({"x": 1})
    with pytest.raises(RuntimeError):
        p.get_optim_params()


def test_reset_clears_queue_and_resets_coreai():
    fake = _FakeCoreAI()
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=fake)
    p.select_action({"observation.state": [0.0] * 7})
    p.reset()
    assert len(p._queue) == 0
    assert fake.resets == 1


def test_processor_factory_returns_real_pipelines():
    pre, post = make_coreai_bridge_pre_post_processors(
        CoreAIBridgeConfig(runtime_binding_mode="in_memory"))
    # v1.3.6: real PolicyProcessorPipeline (not custom identity callables).
    # (PolicyProcessorPipeline is a subscripted generic, so check by capability.)
    assert type(pre).__name__ == "DataProcessorPipeline"
    assert hasattr(pre, "save_pretrained") and hasattr(pre, "from_pretrained")
    assert list(pre.steps) == [] and list(post.steps) == []
    # Step-empty pipelines preserve observation content and action tensors.
    batch = {"observation.state": torch.zeros(1, 7), "task": ["pick"]}
    out = pre(batch)
    assert torch.equal(out["observation.state"], batch["observation.state"])
    assert out["task"] == ["pick"]
    act = torch.ones(1, 7)
    assert torch.equal(post(act), act)


def test_config_does_not_support_training():
    with pytest.raises(NotImplementedError):
        CoreAIBridgeConfig(runtime_binding_mode="in_memory").get_optimizer_preset()
