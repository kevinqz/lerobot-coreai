# test_action_blocker.py — tests for the ActionBlocker shadow-mode invariant.

import pytest
import dataclasses

from lerobot_coreai.action_blocker import ActionBlocker, BlockedAction
from lerobot_coreai.errors import SafetyError


class TestActionBlocker:
    def test_block_returns_blocked_action_not_sent(self):
        b = ActionBlocker()
        result = b.block([0.0] * 7)
        assert isinstance(result, BlockedAction)
        assert result.sent is False
        assert result.destination == "none"
        assert result.mode == "shadow"

    def test_blocked_count_increments(self):
        b = ActionBlocker()
        b.block([0.0] * 7)
        b.block([0.1] * 7)
        b.block([0.2] * 7)
        assert b.blocked_count == 3

    def test_actions_sent_always_zero(self):
        b = ActionBlocker()
        b.block([0.0] * 7)
        b.block([0.1] * 7)
        assert b.actions_sent == 0
        assert b.sent_count == 0

    def test_send_always_raises_safety_error(self):
        b = ActionBlocker()
        with pytest.raises(SafetyError, match="No robot commands were sent"):
            b.send([0.0] * 7)

    def test_send_raises_even_after_blocks(self):
        b = ActionBlocker()
        b.block([0.0] * 7)
        b.block([0.1] * 7)
        with pytest.raises(SafetyError):
            b.send([0.2] * 7)
        assert b.actions_sent == 0

    def test_destination_always_none(self):
        b = ActionBlocker()
        result = b.block([0.0] * 7)
        assert result.destination == "none"

    def test_blocked_action_is_frozen(self):
        """BlockedAction is a frozen dataclass — it must be immutable."""
        b = ActionBlocker()
        result = b.block([0.0] * 7)
        assert dataclasses.is_dataclass(result)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.sent = True  # type: ignore[misc]

    def test_custom_reason_recorded(self):
        b = ActionBlocker()
        b.block([0.0] * 7, reason="test_reason")
        assert "test_reason" in b.reasons
        result = b.block([0.1] * 7, reason="another_reason")
        assert result.reason == "another_reason"

    def test_default_reason(self):
        b = ActionBlocker()
        result = b.block([0.0] * 7)
        assert result.reason == "shadow_mode_no_actuation"
