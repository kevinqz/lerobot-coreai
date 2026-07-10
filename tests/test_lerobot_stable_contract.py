# test_lerobot_stable_contract.py — stable-target contract (v1.2.4).
# Meaningful when LeRobot is installed (the stable CI job installs 0.6.0).

import pytest

from lerobot_coreai.lerobot_contracts import evaluate_compatibility_contract


def _has_lerobot():
    try:
        import importlib.metadata as md
        md.version("lerobot")
        return True
    except Exception:
        return False


def test_stable_strict_requires_installed_lerobot():
    report = evaluate_compatibility_contract(strict=True)
    if not _has_lerobot():
        # Strict without LeRobot: stable target cannot pass.
        assert report["targets"]["stable"]["passed"] is False
        assert report["ok"] is False


@pytest.mark.skipif(not _has_lerobot(), reason="lerobot not installed")
def test_stable_target_passes_when_in_range():
    report = evaluate_compatibility_contract(strict=True)
    assert report["targets"]["stable"]["passed"] is True
    assert report["levels"]["lerobot_version_supported"] == "passed"
    assert report["levels"]["dataset_constructor"] == "passed"
    # Even with LeRobot present, official levels stay honestly failed.
    assert report["levels"]["official_eval"] == "failed"


@pytest.mark.skipif(not _has_lerobot(), reason="lerobot not installed")
def test_detections_populated_with_lerobot():
    report = evaluate_compatibility_contract(strict=False)
    d = report["detections"]
    assert d["lerobot_installed"] is True
    assert d["lerobot_version"] is not None
    assert d["plugin_discovery_mechanism"] in ("detected", "absent")
