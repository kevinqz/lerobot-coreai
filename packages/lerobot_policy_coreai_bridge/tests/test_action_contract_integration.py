# test_action_contract_integration.py — plugin uses the manifest action contract (v1.3.4).

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lerobot_coreai.errors import CoreAIPolicyError  # noqa: E402
from lerobot_policy_coreai_bridge import CoreAIBridgeConfig, CoreAIBridgePolicy  # noqa: E402


class _Fake:
    """Fake CoreAI policy with a declared action contract and a fixed chunk."""

    def __init__(self, representation="chunk", horizon=3, action_dim=7, chunk=None):
        self.robot_type = "so100"
        self.manifest = {"contracts": {"action": {
            "representation": representation, "horizon": horizon,
            "action_dim": action_dim}}}
        if chunk is not None:
            self._chunk = chunk
        elif representation == "single":
            self._chunk = [0.0] * action_dim            # single [A]
        else:
            self._chunk = [[float(i)] * action_dim for i in range(horizon)]

    def predict_action_chunk(self, batch, **kw):
        return self._chunk

    def reset(self):
        pass


def _obs():
    return {"observation.state": torch.zeros(1, 7)}


def test_chunk_contract_respected():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=_Fake(horizon=3))
    assert p.predict_action_chunk(_obs()).shape == (1, 3, 7)


def test_single_contract_respected():
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"),
                           coreai_policy=_Fake(representation="single", horizon=1))
    # A single-action policy returns [A]; the plugin normalizes to [1,1,A] then
    # select_action yields [1,A].
    assert p.select_action(_obs()).shape == (1, 7)


def test_wrong_horizon_from_runner_fails():
    # Manifest says horizon 3, but the runner returns 8 rows -> fail closed.
    fake = _Fake(horizon=3, chunk=[[0.0] * 7 for _ in range(8)])
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=fake)
    with pytest.raises(CoreAIPolicyError):
        p.predict_action_chunk(_obs())


def test_wrong_action_dim_from_runner_fails():
    fake = _Fake(horizon=3, action_dim=7, chunk=[[0.0] * 9 for _ in range(3)])
    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=fake)
    with pytest.raises(CoreAIPolicyError):
        p.predict_action_chunk(_obs())


def test_runner_options_carry_encoding_and_hash():
    captured = {}

    class _Capture(_Fake):
        def predict_action_chunk(self, batch, runner_options=None, **kw):
            captured.update(runner_options or {})
            return self._chunk

    p = CoreAIBridgePolicy(CoreAIBridgeConfig(runtime_binding_mode="in_memory"), coreai_policy=_Capture(horizon=3))
    p.predict_action_chunk(_obs())
    assert captured["observation_encoding"] == "nested_json_v1"
    assert captured["protocol_version"] == "coreai-runner.v2"
    assert captured["observation_sha256"].startswith("sha256:")
