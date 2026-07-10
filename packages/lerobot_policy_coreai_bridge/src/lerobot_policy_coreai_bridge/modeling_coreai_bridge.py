# modeling_coreai_bridge.py — official LeRobot policy for the CoreAI bridge.
#
# A real PreTrainedPolicy (hence torch.nn.Module) that forwards inference to a
# CoreAI-backed policy and returns LeRobot-correct actions: select_action gives
# one per-timestep action as a torch.Tensor(B, action_dim), filled from a chunk
# queue. It is RUNTIME-ONLY: forward()/get_optim_params() raise, train(True)
# raises, eval()/train(False) work. No hardware, no egress from this class.

from __future__ import annotations

from collections import deque
from typing import Any

import torch

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig


def _as_batched_tensor(action: Any) -> torch.Tensor:
    """Coerce a per-timestep action into a (B, action_dim) float tensor."""
    t = torch.as_tensor(action, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)  # add batch dim
    return t


class CoreAIBridgePolicy(PreTrainedPolicy):
    """LeRobot-shaped policy backed by a CoreAI runtime. Runtime-only."""

    config_class = CoreAIBridgeConfig
    name = POLICY_TYPE

    def __init__(self, config: CoreAIBridgeConfig, coreai_policy: Any = None):
        super().__init__(config)
        self.config = config
        self.coreai_policy = coreai_policy
        self._queue: deque = deque()
        # A non-persistent sentinel buffer so .to(device)/.parameters() behave.
        self.register_buffer("_sentinel", torch.zeros(1), persistent=False)

    # MARK: - Inference (LeRobot contract)

    def reset(self) -> None:
        self._queue.clear()
        if self.coreai_policy is not None and hasattr(self.coreai_policy, "reset"):
            self.coreai_policy.reset()

    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        if self.coreai_policy is None:
            raise RuntimeError("coreai_bridge has no CoreAI policy bound.")
        chunk = self.coreai_policy.predict_action_chunk(batch)
        return torch.as_tensor(chunk, dtype=torch.float32)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        if not self._queue:
            chunk = self.predict_action_chunk(batch, **kwargs)
            # chunk is [H, A]; enqueue each timestep row.
            rows = chunk if chunk.ndim == 2 else chunk.unsqueeze(0)
            for row in rows:
                self._queue.append(row)
        action = self._queue.popleft()
        return _as_batched_tensor(action)

    # MARK: - Training boundary (runtime-only)

    def forward(self, batch: dict[str, Any]) -> Any:
        raise RuntimeError(
            "coreai_bridge is runtime-only; it has no training forward. "
            "Train with LeRobot, run with CoreAI.")

    def get_optim_params(self) -> Any:
        raise RuntimeError("coreai_bridge is runtime-only; it exposes no optimizer params.")

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError(
                "coreai_bridge is runtime-only; it cannot enter training mode.")
        return super().train(False)
