# test_profile_calibration.py — profile calibration from actions (v0.9.1).

import json
from importlib.resources import files

import jsonschema
import pytest

from lerobot_coreai.errors import CoreAIPolicyError
from lerobot_coreai.profile_calibration import (
    ProfileCalibrationConfig,
    build_calibration_report,
    calibrate_profile,
    compute_action_statistics,
)
from lerobot_coreai.safety_profiles import load_builtin_profile


def _write_actions(tmp_path, n=100, val=0.5, shape=(16, 7)):
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            act = [[val] * shape[1] for _ in range(shape[0])] if len(shape) == 2 \
                else [val] * shape[0]
            f.write(json.dumps({"step": i, "ok": True, "action": act}) + "\n")
    return path


def test_statistics_extraction(tmp_path):
    path = _write_actions(tmp_path, n=50, val=0.5)
    stats = compute_action_statistics(path)
    assert stats["valid_actions"] == 50
    assert stats["invalid_actions"] == 0
    assert stats["dominant_shape"] == [16, 7]
    assert stats["abs"]["p995"] == pytest.approx(0.5, abs=1e-6)


def test_invalid_and_nan_counted(tmp_path):
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"action": [[0.1] * 7] * 16}) + "\n")
        f.write(json.dumps({"action": None}) + "\n")           # invalid
        f.write(json.dumps({"action": [[float("nan")] * 7] * 16}) + "\n")  # nan
        f.write("not json\n")                                   # invalid
    stats = compute_action_statistics(path)
    assert stats["invalid_actions"] == 2
    assert stats["nan_action_steps"] == 1


def test_calibration_uses_quantile_and_margin(tmp_path):
    path = _write_actions(tmp_path, n=100, val=0.5)
    base = load_builtin_profile("so100-sim-default")
    result = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path, base_profile=base,
        quantile=0.995, margin=0.10))
    # p995 abs ~0.5, *1.1 = 0.55.
    assert result.profile.max_abs_action == pytest.approx(0.55, abs=1e-3)
    assert result.samples == 100


def test_conservative_does_not_exceed_base(tmp_path):
    path = _write_actions(tmp_path, n=100, val=0.95)  # near base bound
    base = load_builtin_profile("so100-sim-default")  # max_abs 1.0
    result = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path, base_profile=base,
        margin=0.5, conservative=True))
    # 0.95*1.5 = 1.425 > base 1.0 → conservative clamps to base.
    assert result.profile.max_abs_action <= base.max_abs_action


def _write_actions_with_outliers(tmp_path, n=100):
    # Most values small (0.1); a few large outliers (0.9) so p95 < p995.
    path = tmp_path / "actions.jsonl"
    with open(path, "w") as f:
        for i in range(n):
            v = 0.9 if i % 50 == 0 else 0.1   # ~2% outliers
            f.write(json.dumps({"step": i, "action": [[v] * 7 for _ in range(16)]}) + "\n")
    return path


def test_calibration_respects_quantile(tmp_path):
    path = _write_actions_with_outliers(tmp_path, n=200)
    r95 = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path, quantile=0.95, margin=0.0))
    r995 = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path, quantile=0.995, margin=0.0))
    # p95 excludes the outliers; p995 includes them → smaller bound at p95.
    assert r95.profile.max_abs_action < r995.profile.max_abs_action
    assert r95.report["quantile"] == 0.95
    assert r95.report["quantile_key"] == "p95"


def test_unsupported_quantile_fails_clear(tmp_path):
    path = _write_actions(tmp_path, n=50)
    with pytest.raises(CoreAIPolicyError, match="Unsupported calibration quantile"):
        calibrate_profile(ProfileCalibrationConfig(
            actions_path=path, output_dir=tmp_path, quantile=0.987))


def test_calibration_method_reflects_quantile(tmp_path):
    path = _write_actions(tmp_path, n=50)
    result = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path, quantile=0.99, margin=0.10))
    assert "quantile_0.99" in result.profile.calibration_method


def test_insufficient_samples_fails(tmp_path):
    path = _write_actions(tmp_path, n=3)
    with pytest.raises(CoreAIPolicyError, match="Insufficient samples"):
        calibrate_profile(ProfileCalibrationConfig(
            actions_path=path, output_dir=tmp_path, min_samples=10))


def test_generated_profile_validates_schema(tmp_path):
    path = _write_actions(tmp_path, n=100, val=0.4)
    base = load_builtin_profile("so100-sim-default")
    result = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path, base_profile=base))
    schema = json.loads(
        files("lerobot_coreai.schemas").joinpath("safety-profile.schema.json").read_text())
    jsonschema.validate(result.profile.to_dict(), schema)


def test_report_has_honest_claims(tmp_path):
    path = _write_actions(tmp_path, n=50)
    result = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path))
    claims = result.report["claims"]
    assert claims["proves_future_action_safety"] is False
    assert claims["proves_physical_safety"] is False
    assert claims["proves_real_world_safety"] is False


def test_calibration_report_validates_schema(tmp_path):
    path = _write_actions(tmp_path, n=50)
    result = calibrate_profile(ProfileCalibrationConfig(
        actions_path=path, output_dir=tmp_path))
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "profile-calibration-report.schema.json").read_text())
    jsonschema.validate(result.report, schema)
