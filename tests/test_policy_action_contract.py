# test_policy_action_contract.py — CoreAIPolicy chunk/next-action semantics (v1.2.5).

from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.policy import CoreAIPolicy


def _policy(valid_manifest_dict, chunk=None):
    m = LeRobotCoreAIManifest.from_dict(valid_manifest_dict)
    p = CoreAIPolicy(m, validate_io=False)
    canned = chunk if chunk is not None else [[float(i)] * 7 for i in range(16)]
    p.predict_action = lambda batch, **kw: {"action": canned, "metadata": {}}  # type: ignore
    return p, canned


def test_predict_action_chunk_returns_full_chunk(valid_manifest_dict):
    p, canned = _policy(valid_manifest_dict)
    assert p.predict_action_chunk({"observation.state": [0.0] * 7}) == canned


def test_legacy_select_action_still_returns_chunk(valid_manifest_dict):
    # Backward compatibility: select_action still returns the whole chunk.
    p, canned = _policy(valid_manifest_dict)
    assert p.select_action({"observation.state": [0.0] * 7}) == canned


def test_select_next_action_returns_single_row_and_drains(valid_manifest_dict):
    chunk = [[1.0] * 7, [2.0] * 7, [3.0] * 7]
    p, _ = _policy(valid_manifest_dict, chunk=chunk)
    batch = {"observation.state": [0.0] * 7}
    assert p.select_next_action(batch) == [1.0] * 7
    assert p.select_next_action(batch) == [2.0] * 7
    assert p.select_next_action(batch) == [3.0] * 7
    # Queue exhausted → refills from a fresh chunk (same canned chunk here).
    assert p.select_next_action(batch) == [1.0] * 7


def test_reset_clears_queue(valid_manifest_dict):
    chunk = [[1.0] * 7, [2.0] * 7]
    p, _ = _policy(valid_manifest_dict, chunk=chunk)
    batch = {"observation.state": [0.0] * 7}
    assert p.select_next_action(batch) == [1.0] * 7
    assert len(p._action_queue) == 1
    p.reset()
    assert p._action_queue.empty is True
    # After reset, next call refills and starts from the top.
    assert p.select_next_action(batch) == [1.0] * 7


def test_action_contract_inferred_on_policy(valid_manifest_dict):
    p, _ = _policy(valid_manifest_dict)
    assert p._action_contract.representation == "chunk"
    assert p._action_contract.horizon == 16
