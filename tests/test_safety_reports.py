# test_safety_reports.py — safety report/summary builders (v0.9.0).

import json

from lerobot_coreai.safety_reports import (
    SafetyAccumulator,
    append_safety_decision,
    build_safety_summary,
    build_safety_summary_markdown,
    decision_record,
)
from lerobot_coreai.safety_supervisor import SafetyContext, SafetyDecision


def _decision(allowed=True, modified=False, severity="info", reasons=None):
    return SafetyDecision(
        allowed=allowed, action_modified=modified, reasons=reasons or [],
        checks=[{"name": "finite", "passed": allowed, "severity": "critical"}],
        profile="p", mode="enforce", severity=severity,
    )


def test_decision_record_shape():
    ctx = SafetyContext(mode="sim", episode=1, step=2)
    rec = decision_record(_decision(), context=ctx)
    assert rec["episode"] == 1
    assert rec["step"] == 2
    assert rec["mode"] == "sim"
    assert "checks" in rec and "reasons" in rec


def test_append_safety_decision_jsonl(tmp_path):
    path = tmp_path / "safety_report.jsonl"
    append_safety_decision(path, _decision(), context=SafetyContext(mode="sim", step=0))
    append_safety_decision(path, _decision(allowed=False, severity="critical",
                                           reasons=["finite"]),
                           context=SafetyContext(mode="sim", step=1))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["allowed"] is False


def test_accumulator_counts():
    acc = SafetyAccumulator(profile="p", mode="enforce")
    acc.add(_decision(allowed=True))
    acc.add(_decision(allowed=True, modified=True, severity="warning",
                      reasons=["action_clipped_to_bounds"]))
    acc.add(_decision(allowed=False, severity="critical", reasons=["finite"]))
    assert acc.actions_supervised == 3
    assert acc.actions_allowed == 2
    assert acc.actions_blocked == 1
    assert acc.actions_modified == 1
    assert acc.critical_failures == 1
    assert acc.passed is False
    assert acc.top_reasons()["finite"] == 1


def test_accumulator_passed_when_clean():
    acc = SafetyAccumulator(profile="p", mode="enforce")
    acc.add(_decision(allowed=True))
    assert acc.passed is True


def test_report_only_would_block_is_not_masked():
    # A report_only unsafe action is allowed=True but must fail the summary.
    from lerobot_coreai.safety_profiles import SafetyProfile
    from lerobot_coreai.safety_supervisor import MODE_REPORT_ONLY, SafetySupervisor

    profile = SafetyProfile(name="t", require_robot_type_match=False,
                            require_known_shape=False)
    sup = SafetySupervisor(profile, mode=MODE_REPORT_ONLY)
    acc = SafetyAccumulator(profile=profile.name, mode=MODE_REPORT_ONLY)
    decision = sup.evaluate([float("nan")]).decision
    acc.add(decision)
    summary = build_safety_summary(acc)

    assert decision.allowed is True
    assert "report_only_would_block" in decision.reasons
    assert summary["actions_blocked"] == 0
    assert summary["would_block_actions"] == 1
    assert summary["critical_findings"] == 1
    assert summary["passed"] is False


def test_clean_run_passes():
    from lerobot_coreai.safety_profiles import SafetyProfile
    from lerobot_coreai.safety_supervisor import MODE_ENFORCE, SafetySupervisor

    profile = SafetyProfile(name="t", require_robot_type_match=False,
                            require_known_shape=False)
    sup = SafetySupervisor(profile, mode=MODE_ENFORCE)
    acc = SafetyAccumulator(profile=profile.name, mode=MODE_ENFORCE)
    acc.add(sup.evaluate([0.0, 0.0]).decision)
    summary = build_safety_summary(acc)
    assert summary["passed"] is True
    assert summary["would_block_actions"] == 0
    assert summary["critical_findings"] == 0


def test_build_summary_has_claims():
    acc = SafetyAccumulator(profile="so100-sim-default", mode="enforce")
    acc.add(_decision(allowed=False, severity="critical", reasons=["finite"]))
    summary = build_safety_summary(acc)
    assert summary["schema_version"] == "lerobot-coreai.safety_summary.v0"
    assert summary["profile"] == "so100-sim-default"
    assert summary["actions_blocked"] == 1
    assert summary["claims"]["proves_physical_safety"] is False
    assert summary["claims"]["proves_real_world_safety"] is False
    assert summary["claims"]["proves_software_supervision"] is True


def test_summary_markdown_no_overclaim():
    acc = SafetyAccumulator(profile="p", mode="enforce")
    acc.add(_decision())
    md = build_safety_summary_markdown(build_safety_summary(acc))
    assert "does not prove physical robot safety" in md
    lower = md.lower()
    assert "proves physical safety: true" not in lower
