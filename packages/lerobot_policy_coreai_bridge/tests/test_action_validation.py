# test_action_validation.py — strict Runner-output normalization (v1.3.3).

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lerobot_coreai.errors import CoreAIPolicyError  # noqa: E402
from lerobot_policy_coreai_bridge.action_validation import (  # noqa: E402
    normalize_and_validate_action_chunk as norm,
)


def test_chunk_hw_a_normalized():
    t = norm([[1.0] * 7] * 3, representation="chunk", expected_action_dim=7)
    assert t.shape == (1, 3, 7)


def test_single_a_normalized():
    t = norm([1.0] * 7, representation="single", expected_action_dim=7)
    assert t.shape == (1, 1, 7)


def test_wrong_action_dim_fails():
    with pytest.raises(CoreAIPolicyError):
        norm([[1.0] * 5] * 3, representation="chunk", expected_action_dim=7)


def test_wrong_horizon_fails():
    with pytest.raises(CoreAIPolicyError):
        norm([[1.0] * 7] * 3, representation="chunk", expected_horizon=16)


def test_rank_4_fails():
    with pytest.raises(CoreAIPolicyError):
        norm([[[[1.0] * 7]]], representation="chunk")


def test_non_finite_fails():
    with pytest.raises(CoreAIPolicyError):
        norm([[1.0, float("nan")] + [0.0] * 5], representation="chunk")


def test_device_honored():
    t = norm([[1.0] * 7], representation="chunk", device=torch.device("cpu"))
    assert t.device == torch.device("cpu")


def test_ragged_fails():
    with pytest.raises(CoreAIPolicyError):
        norm([[1.0, 2.0], [3.0]], representation="chunk")
