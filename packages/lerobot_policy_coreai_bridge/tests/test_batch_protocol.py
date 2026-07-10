# test_batch_protocol.py — batch execution decision (v1.3.9).

from dataclasses import dataclass

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_policy_coreai_bridge.batch_protocol import (
    MODE_NATIVE, MODE_SINGLE, MODE_SPLIT, select_batch_execution_mode,
)


@dataclass
class _Cfg:
    batch_mode: str = "auto"
    max_batch_size: int | None = None


@dataclass
class _Contract:
    max_batch_size: int = 8


@dataclass
class _Caps:
    supports_batch: bool = False
    max_batch_size: int | None = None
    action_batching_semantics: str | None = None
    action_batching_state_isolation: str | None = None
    inference_state_scope: str | None = None
    supports_session_ids: bool = False


def _native_caps(**kw):
    base = dict(supports_batch=True, max_batch_size=4, action_batching_semantics="native",
                action_batching_state_isolation="stateless",
                inference_state_scope="stateless")
    base.update(kw)
    return _Caps(**base)


# --- B=1 is always safe ---

def test_b1_single_regardless_of_caps():
    d = select_batch_execution_mode(_Cfg("auto"), None, None, 1)
    assert d.mode == MODE_SINGLE


def test_single_only_rejects_b_gt_1():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("single_only"), _Contract(), _native_caps(), 2)


# --- scope safety for B>1 ---

def test_missing_scope_fails_for_batch():
    caps = _native_caps(inference_state_scope=None)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Contract(), caps, 2)


def test_unknown_scope_fails():
    caps = _native_caps(inference_state_scope="weird")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Contract(), caps, 2)


def test_global_scope_forbidden():
    caps = _native_caps(inference_state_scope="global")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Contract(), caps, 2)


def test_session_scoped_deferred():
    caps = _native_caps(inference_state_scope="session_scoped", supports_session_ids=True)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("split_and_stack"), _Contract(), caps, 2)


# --- native ---

def test_native_ok():
    d = select_batch_execution_mode(_Cfg("native_batch"), _Contract(), _native_caps(), 4)
    assert d.mode == MODE_NATIVE and d.batch_size == 4


def test_native_requires_state_isolation():
    caps = _native_caps(action_batching_state_isolation="shared")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Contract(), caps, 2)


def test_native_requires_native_semantics():
    caps = _native_caps(action_batching_semantics="split_and_stack")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Contract(), caps, 2)


# --- split ---

def test_split_stateless_ok():
    caps = _Caps(inference_state_scope="stateless", max_batch_size=4)
    d = select_batch_execution_mode(_Cfg("split_and_stack"), _Contract(), caps, 3)
    assert d.mode == MODE_SPLIT and d.batch_size == 3


def test_split_request_scoped_ok():
    caps = _Caps(inference_state_scope="request_scoped", max_batch_size=4)
    d = select_batch_execution_mode(_Cfg("split_and_stack"), _Contract(), caps, 2)
    assert d.mode == MODE_SPLIT


# --- effective max ---

def test_effective_max_is_minimum():
    # artifact 8, config 3, runner 4 -> effective 3; B=4 must fail.
    caps = _native_caps(max_batch_size=4)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch", max_batch_size=3),
                                    _Contract(max_batch_size=8), caps, 4)
    d = select_batch_execution_mode(_Cfg("native_batch", max_batch_size=3),
                                    _Contract(max_batch_size=8), caps, 3)
    assert d.mode == MODE_NATIVE and d.effective_max_batch_size == 3


def test_batch_above_runner_max_fails():
    caps = _native_caps(max_batch_size=2)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Contract(max_batch_size=8),
                                    caps, 4)


def test_invalid_runner_max_fails():
    caps = _native_caps(max_batch_size=0)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Contract(), caps, 2)


# --- auto ---

def test_auto_prefers_native():
    d = select_batch_execution_mode(_Cfg("auto"), _Contract(), _native_caps(), 2)
    assert d.mode == MODE_NATIVE


def test_auto_split_when_no_native():
    caps = _Caps(inference_state_scope="stateless", max_batch_size=4)
    d = select_batch_execution_mode(_Cfg("auto"), _Contract(), caps, 2)
    assert d.mode == MODE_SPLIT
