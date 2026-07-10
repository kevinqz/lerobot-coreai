# test_obs_bridge.py — observation pipeline bridge check (v1.1.5).

import json
from importlib.resources import files

import jsonschema

from lerobot_coreai.manifest import LeRobotCoreAIManifest
from lerobot_coreai.obs_bridge import OBS_BRIDGE_SCHEMA_VERSION, evaluate_obs_bridge
from lerobot_coreai.observation_adapters import ObservationAdapterConfig


def _manifest(valid_manifest_dict):
    return LeRobotCoreAIManifest.from_dict(valid_manifest_dict)


def _state_key(m):
    # Pick a state-like feature key from the manifest.
    for n in m.observation_features:
        if "state" in n:
            return n
    return next(iter(m.observation_features))


def _full_raw(m):
    raw = {}
    for name, spec in m.observation_features.items():
        if "image" in name:
            raw[name] = "img"
        elif spec.shape is not None:
            raw[name] = [0.0] * int(spec.shape[-1])
        else:
            raw[name] = 0.0
    if "task" in m.observation_features:
        raw["task"] = "do the thing"
    return raw


def test_valid_frame_passes(valid_manifest_dict):
    m = _manifest(valid_manifest_dict)
    cfg = ObservationAdapterConfig(state_key=_state_key(m))
    report = evaluate_obs_bridge(_full_raw(m), cfg, manifest=m, policy_path="p")
    assert report["ok"] is True
    assert report["schema_version"] == OBS_BRIDGE_SCHEMA_VERSION


def test_schema_valid(valid_manifest_dict):
    m = _manifest(valid_manifest_dict)
    cfg = ObservationAdapterConfig(state_key=_state_key(m))
    report = evaluate_obs_bridge(_full_raw(m), cfg, manifest=m)
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "obs-bridge-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_missing_required_key_fails(valid_manifest_dict):
    m = _manifest(valid_manifest_dict)
    sk = _state_key(m)
    raw = _full_raw(m)
    raw.pop(sk, None)  # drop the state
    cfg = ObservationAdapterConfig(state_key=sk, require_state=True)
    report = evaluate_obs_bridge(raw, cfg, manifest=m)
    assert report["ok"] is False
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["required_keys_present"] is False


def test_state_shape_mismatch_detected(valid_manifest_dict):
    m = _manifest(valid_manifest_dict)
    sk = _state_key(m)
    spec = m.observation_features[sk]
    if spec.shape is None:
        return  # no shape constraint to violate
    raw = _full_raw(m)
    raw[sk] = [0.0] * (int(spec.shape[-1]) + 3)  # wrong length
    cfg = ObservationAdapterConfig(state_key=sk)
    report = evaluate_obs_bridge(raw, cfg, manifest=m)
    names = {c["name"]: c["passed"] for c in report["checks"]}
    assert names["state_shape_compatible"] is False


def test_claims_never_task_success(valid_manifest_dict):
    m = _manifest(valid_manifest_dict)
    cfg = ObservationAdapterConfig(state_key=_state_key(m))
    report = evaluate_obs_bridge(_full_raw(m), cfg, manifest=m)
    assert report["claims"]["proves_task_success"] is False
    assert report["claims"]["proves_physical_safety"] is False


def test_no_silent_drop_reported(valid_manifest_dict):
    m = _manifest(valid_manifest_dict)
    raw = _full_raw(m)
    raw["observation.unexpected"] = [1, 2, 3]
    cfg = ObservationAdapterConfig(state_key=_state_key(m), drop_unknown_keys=True)
    report = evaluate_obs_bridge(raw, cfg, manifest=m)
    drop = next(c for c in report["checks"] if c["name"] == "no_silent_drop")
    assert "observation.unexpected" in drop["detail"]
    assert "observation.unexpected" in report["dropped_keys"]
