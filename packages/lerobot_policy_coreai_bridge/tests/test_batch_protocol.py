# test_batch_protocol.py — batch execution decision foundation (v1.3.8, no B>1).

from dataclasses import dataclass

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_policy_coreai_bridge.batch_protocol import (
    MODE_NATIVE, MODE_SINGLE, MODE_SPLIT, select_batch_execution_mode,
)


@dataclass
class _Cfg:
    batch_mode: str = "auto"


@dataclass
class _Caps:
    supports_batch: bool = False
    max_batch_size: int = 1
    action_batching_semantics: str | None = None
    inference_state_scope: str | None = None
    supports_session_ids: bool = False


def test_single_only_stays_single():
    d = select_batch_execution_mode(_Cfg("single_only"), _Caps())
    assert d.mode == MODE_SINGLE


def test_native_requires_native_support():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Caps(supports_batch=False))


def test_native_ok_when_announced():
    caps = _Caps(supports_batch=True, action_batching_semantics="native", max_batch_size=4)
    d = select_batch_execution_mode(_Cfg("native_batch"), caps)
    assert d.mode == MODE_NATIVE and d.max_batch_size == 4


def test_split_allowed_stateless():
    caps = _Caps(inference_state_scope="stateless", max_batch_size=4)
    d = select_batch_execution_mode(_Cfg("split_and_stack"), caps)
    assert d.mode == MODE_SPLIT and not d.requires_session_ids


def test_split_global_forbidden():
    caps = _Caps(inference_state_scope="global")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("split_and_stack"), caps)


def test_split_session_scoped_requires_session_ids():
    caps = _Caps(inference_state_scope="session_scoped", supports_session_ids=False)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("split_and_stack"), caps)
    caps2 = _Caps(inference_state_scope="session_scoped", supports_session_ids=True,
                  max_batch_size=2)
    d = select_batch_execution_mode(_Cfg("split_and_stack"), caps2)
    assert d.mode == MODE_SPLIT and d.requires_session_ids


def test_unknown_scope_fails():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Caps(inference_state_scope="bogus"))


def test_auto_prefers_native():
    caps = _Caps(supports_batch=True, action_batching_semantics="native", max_batch_size=8)
    assert select_batch_execution_mode(_Cfg("auto"), caps).mode == MODE_NATIVE


def test_auto_falls_back_to_split_then_single():
    split = select_batch_execution_mode(_Cfg("auto"), _Caps(inference_state_scope="stateless"))
    assert split.mode == MODE_SPLIT
    single = select_batch_execution_mode(_Cfg("auto"), _Caps(inference_state_scope="global"))
    assert single.mode == MODE_SINGLE
