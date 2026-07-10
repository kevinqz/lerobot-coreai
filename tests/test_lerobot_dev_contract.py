# test_lerobot_dev_contract.py — development-target contract metadata (v1.2.4).
# The development target is a pinned, non-blocking probe. These assertions hold
# regardless of which LeRobot (if any) is installed.

from lerobot_coreai.lerobot_contracts import (
    DEVELOPMENT_TARGET_VERSION, evaluate_compatibility_contract,
)


def test_development_target_declared_non_blocking():
    report = evaluate_compatibility_contract(strict=False)
    dev = report["targets"]["development"]
    assert dev["version"] == DEVELOPMENT_TARGET_VERSION
    assert dev["required"] is False


def test_development_does_not_force_official_claims_true():
    # A dev LeRobot must never flip official-support claims to true.
    report = evaluate_compatibility_contract(strict=False)
    assert report["claims"]["official_eval_compatible"] is False
    assert report["claims"]["official_plugin_compatible"] is False


def test_levels_are_valid_outcomes():
    report = evaluate_compatibility_contract(strict=False)
    allowed = {"passed", "partial", "failed", "not_tested", "not_supported",
               "separate_runtime"}
    for v in report["levels"].values():
        assert v in allowed
