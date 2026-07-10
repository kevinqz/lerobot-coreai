# test_batch_protocol.py — authoritative batch execution decision (v1.3.10).

from dataclasses import dataclass, field

import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_policy_coreai_bridge.batch_protocol import (
    MODE_NATIVE, MODE_SINGLE, MODE_SPLIT, select_batch_execution_mode,
)


@dataclass
class _Cfg:
    batch_mode: str = "auto"
    max_batch_size: int | None = None
    max_split_requests: int | None = None


@dataclass
class _Contract:
    authoritative: bool = True
    native_supported: bool = True
    native_max_batch_size: int = 4
    native_slot_isolation: str = "independent"
    split_supported: bool = True
    split_max_batch_size: int = 4
    split_allowed_scopes: tuple = ("stateless", "request_scoped")
    fallback: str = "split_and_stack"
    queue_layout: str = "time_major_batched"
    commit_semantics: str = "atomic_queue_commit"


@dataclass
class _Caps:
    supports_batch: bool = False
    max_batch_size: int | None = None
    action_batching_semantics: str | None = None
    action_batching_slot_isolation: str | None = None
    action_batching_state_isolation: str | None = None
    inference_state_scope: str | None = None


def _native_caps(**kw):
    base = dict(supports_batch=True, max_batch_size=4, action_batching_semantics="native",
                action_batching_slot_isolation="independent",
                inference_state_scope="stateless")
    base.update(kw)
    return _Caps(**base)


# --- B=1 ---

def test_b1_single_regardless():
    assert select_batch_execution_mode(_Cfg("auto"), None, None, 1).mode == MODE_SINGLE


def test_single_only_rejects_batch():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("single_only"), _Contract(), _native_caps(), 2)


# --- contract authority (P1.1) ---

def test_non_authoritative_contract_rejects_batch():
    c = _Contract(authoritative=False)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), c, _native_caps(), 2)


def test_missing_contract_rejects_batch():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), None, _native_caps(), 2)


def test_native_unsupported_in_contract_fails():
    c = _Contract(native_supported=False)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), c, _native_caps(), 2)


def test_split_unsupported_in_contract_fails():
    c = _Contract(split_supported=False)
    caps = _Caps(inference_state_scope="stateless")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("split_and_stack"), c, caps, 2)


def test_fallback_reject_blocks_auto_split():
    c = _Contract(fallback="reject")
    caps = _Caps(inference_state_scope="stateless")   # no native
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), c, caps, 2)


def test_bad_queue_layout_fails():
    c = _Contract(queue_layout="something_else")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), c, _native_caps(), 2)


# --- scope + slot isolation (P1.2/P1.4) ---

def test_missing_scope_fails():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Contract(),
                                    _native_caps(inference_state_scope=None), 2)


def test_global_scope_fails():
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("auto"), _Contract(),
                                    _native_caps(inference_state_scope="global"), 2)


def test_session_scoped_deferred():
    caps = _native_caps(inference_state_scope="session_scoped")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Contract(), caps, 2)


def test_native_requires_independent_slot_isolation():
    caps = _native_caps(action_batching_slot_isolation="shared")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Contract(), caps, 2)


def test_native_ok():
    d = select_batch_execution_mode(_Cfg("native_batch"), _Contract(), _native_caps(), 4)
    assert d.mode == MODE_NATIVE and d.batch_size == 4


# --- separate limits (P1.3) ---

def test_split_not_capped_by_runner_native_max():
    # runner native max=1 (no native), but split B=4 must still be allowed.
    c = _Contract(native_supported=False, split_max_batch_size=8)
    caps = _Caps(inference_state_scope="stateless", supports_batch=False, max_batch_size=1)
    d = select_batch_execution_mode(_Cfg("split_and_stack", max_split_requests=4), c, caps, 4)
    assert d.mode == MODE_SPLIT and d.batch_size == 4


def test_native_effective_max_uses_runner_native_max():
    caps = _native_caps(max_batch_size=2)
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("native_batch"), _Contract(native_max_batch_size=8),
                                    caps, 4)


def test_split_capped_by_max_split_requests():
    c = _Contract(native_supported=False, split_max_batch_size=8)
    caps = _Caps(inference_state_scope="stateless")
    with pytest.raises(CoreAIPolicyError):
        select_batch_execution_mode(_Cfg("split_and_stack", max_split_requests=2), c, caps, 4)


# --- auto ---

def test_auto_prefers_native():
    assert select_batch_execution_mode(_Cfg("auto"), _Contract(), _native_caps(), 2).mode == MODE_NATIVE


def test_auto_split_when_no_native():
    c = _Contract(native_supported=False)
    caps = _Caps(inference_state_scope="stateless")
    assert select_batch_execution_mode(_Cfg("auto"), c, caps, 2).mode == MODE_SPLIT
