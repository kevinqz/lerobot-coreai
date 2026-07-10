# policy_card_templates.py — deterministic policy-card section builders (v1.2.3).
#
# Each builder takes already-loaded, already-verified report data and returns a
# markdown section. No wall-clock, no randomness — the same evidence always
# renders the same card. Mandatory non-claims are never omitted.

from __future__ import annotations

from typing import Any

# The non-claims that must appear on every card, verbatim.
MANDATORY_NON_CLAIMS = [
    "This does not prove physical safety.",
    "This does not prove real-world task success.",
    "This does not authorize unrestricted robot actuation.",
    "This is not upstream-native LeRobot registry integration.",
    "This does not support training inside lerobot-coreai.",
]


def _bool(v) -> str:
    return "yes" if v is True else ("no" if v is False else "unknown")


def section_what(policy_path: str | None, robot_type: str | None) -> str:
    return (
        "## What this policy is\n\n"
        f"- Policy: `{policy_path or 'unknown'}`\n"
        f"- Robot type: `{robot_type or 'unknown'}`\n"
        "- A LeRobot-shaped policy run through the Apple CoreAI runtime "
        "(`lerobot-coreai`).\n")


def section_runtime() -> str:
    return (
        "## Runtime & ecosystem\n\n"
        "- Runtime: **CoreAI** (`.aimodel` via coreai-runner).\n"
        "- Base ecosystem: **LeRobot 0.6.x-shaped** (not upstream-native).\n"
        "- Training boundary: train with **LeRobot**; run with **CoreAI**. "
        "Training is not supported inside `lerobot-coreai`.\n")


def section_how_to_run() -> str:
    return (
        "## How to run (safely)\n\n"
        "```python\n"
        "from lerobot_coreai.lerobot_bridge import load_coreai_policy_for_lerobot\n"
        "policy = load_coreai_policy_for_lerobot(POLICY_PATH, runner_url=RUNNER_URL)\n"
        "action = policy.select_action(batch)\n"
        "```\n\n"
        "Real-hardware egress is only ever available via `real --mode guarded`, "
        "behind every gate. Nothing on this card authorizes actuation.\n")


def section_compat(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    lr = data.get("lerobot_version")
    return (
        "## LeRobot compatibility evidence\n\n"
        f"- Compatibility check ok: {_bool(data.get('ok'))}\n"
        f"- LeRobot version tested: `{lr}`\n"
        f"- Shape-compatible with LeRobot 0.6.x: "
        f"{_bool(data.get('claims', {}).get('compatible_with_lerobot_0_6_x_shape'))}\n")


def section_bridge(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    return (
        "## Bridge evidence\n\n"
        f"- Bridge check ok: {_bool(data.get('ok'))}\n"
        f"- policy_type: `{data.get('policy_type', 'coreai_bridge')}`\n"
        "- Local bridge only — `policy_type=\"coreai\"` is not registered upstream.\n")


def section_registry(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    return (
        "## Registry evidence\n\n"
        f"- Local registry check ok: {_bool(data.get('ok'))}\n"
        "- Local, opt-in registry adapter — not upstream registration.\n")


def section_feature_mapping(eval_v2: dict[str, Any] | None) -> str:
    if not eval_v2:
        return ""
    fm = eval_v2.get("feature_mapping", {})
    return (
        "## Feature mapping summary\n\n"
        f"- Strict features: {_bool(eval_v2.get('strict'))}\n"
        f"- Mapping passed: {_bool(fm.get('passed'))}\n"
        f"- Unknown dataset features: {fm.get('unknown_dataset_features', [])}\n")


def section_eval_v2(eval_v2: dict[str, Any] | None) -> str:
    if not eval_v2:
        return ""
    return (
        "## Eval-v2 summary\n\n"
        f"- Dataset: `{eval_v2.get('dataset_repo_id')}`\n"
        f"- Eval ok: {_bool(eval_v2.get('ok'))}\n"
        f"- Frames evaluated: {eval_v2.get('frames_evaluated', 0)}\n"
        "- Proves observation-mapping coherence for the sample only — "
        "not task success.\n")


def section_obs_bridge(data: dict[str, Any] | None) -> str:
    if not data:
        return ""
    return (
        "## Observation pipeline summary\n\n"
        f"- Obs-bridge ok: {_bool(data.get('ok'))}\n"
        f"- Dropped keys: {data.get('dropped_keys', [])}\n"
        "- No silent drops: dropped keys are always listed.\n")


def section_benchmark(manifest: dict[str, Any] | None) -> str:
    if not manifest:
        return ""
    reports = manifest.get("reports", {})
    lines = ["## Benchmark pack summary", "", "- Bundled reports:"]
    for slot in sorted(reports):
        lines.append(f"  - {slot}")
    return "\n".join(lines) + "\n"


def section_trust(*, provenance: dict | None, signature: dict | None,
                  release_check: dict | None, release_channel: str | None) -> str:
    lines = ["## Provenance / signature / release status", ""]
    lines.append(f"- Provenance present: {_bool(provenance is not None)}")
    if signature:
        lines.append(f"- Signed: yes (fingerprint "
                     f"`{signature.get('signer', {}).get('key_fingerprint')}`)")
    else:
        lines.append("- Signed: no (unsigned)")
    if release_check is not None:
        lines.append(f"- Release check ok: {_bool(release_check.get('ok'))}")
    if release_channel:
        lines.append(f"- Release channel: `{release_channel}`")
    return "\n".join(lines) + "\n"


def section_limitations_and_non_claims() -> str:
    lines = ["## Known limitations & safety boundaries", ""]
    lines.append("- Software evidence only; scope is the tested sample/reports.")
    lines.append("- Guarded real egress (if used) is loopback/operator-controlled.")
    lines += ["", "## Non-claims", ""]
    for nc in MANDATORY_NON_CLAIMS:
        lines.append(f"- {nc}")
    return "\n".join(lines) + "\n"
