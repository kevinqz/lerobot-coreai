# test_episode_staging.py — mobile-recording bridge (RFC-0700 §22 / RFC-0900 §22, LR11).
# Validation fails closed; the conversion plan is deterministic. Pure base package.

import pytest

from lerobot_coreai.episode_staging import (
    EPISODE_STAGING_SCHEMA, build_lerobot_conversion_plan, validate_episode_events,
    validate_episode_manifest,
)


def _manifest(**over):
    m = {"schema": EPISODE_STAGING_SCHEMA, "session_id": "S1",
         "created_monotonic_ns": 1000, "fps": 10,
         "features": {
             "observation.state": {"dtype": "float32", "shape": [6], "max_age_ns": 200},
             "image.front": {"dtype": "image", "shape": [3, 8, 8]},
             "action": {"dtype": "float32", "shape": [6]}},
         "events_ref": "events.jsonl", "checksums_ref": "checksums.json"}
    m.update(over)
    return m


def _events():
    out = []
    seq = {"cam": 0, "state": 0, "act": 0}
    for t in range(0, 300, 100):     # 3 aligned frames, state within max_age (200)
        for feat, src in (("observation.state", "state"), ("image.front", "cam"),
                          ("action", "act")):
            seq[src] += 1
            out.append({"session_id": "S1", "monotonic_ns": t, "sequence": seq[src],
                        "source": src, "schema_version": "v1", "feature": feat})
    return out


def test_valid_manifest_and_events():
    m = _manifest()
    ok, errs = validate_episode_manifest(m)
    assert ok, errs
    ok, errs = validate_episode_events(m, _events())
    assert ok, errs


def test_manifest_wrong_schema_rejected():
    ok, errs = validate_episode_manifest(_manifest(schema="something.else"))
    assert not ok and any("schema" in e for e in errs)


def test_session_identity_break_rejected():
    evs = _events(); evs[0]["session_id"] = "OTHER"
    ok, errs = validate_episode_events(_manifest(), evs)
    assert not ok and any("identity break" in e for e in errs)


def test_non_monotonic_time_rejected():
    evs = _events(); evs[-1]["monotonic_ns"] = 0      # goes backwards for its source
    ok, errs = validate_episode_events(_manifest(), evs)
    assert not ok and any("non-monotonic" in e for e in errs)


def test_missing_required_feature_rejected():
    # drop every 'action' event → a declared feature was never recorded.
    evs = [e for e in _events() if e["feature"] != "action"]
    ok, errs = validate_episode_events(_manifest(), evs)
    assert not ok and any("never recorded" in e for e in errs)


def test_freshness_violation_rejected():
    m = _manifest()
    evs = [e for e in _events() if e["feature"] == "observation.state"]
    evs[1]["monotonic_ns"] = 100000        # gap >> max_age_ns (200)
    ok, errs = validate_episode_events(m, evs)
    assert not ok and any("max_age" in e for e in errs)


def test_conversion_plan_maps_to_lerobot_keys():
    plan = build_lerobot_conversion_plan(_manifest())
    assert plan["target"] == "LeRobotDataset" and plan["fps"] == 10
    assert plan["features_map"]["image.front"] == "observation.images.front"
    assert plan["features_map"]["action"] == "action"
    assert plan["features_map"]["observation.state"] == "observation.state"
    assert plan["requires_upstream_lerobot"] is True     # actual write is a rollout step
    assert plan["episodes"] == 1


def test_conversion_plan_refuses_invalid_manifest():
    with pytest.raises(ValueError):
        build_lerobot_conversion_plan(_manifest(features={}))
