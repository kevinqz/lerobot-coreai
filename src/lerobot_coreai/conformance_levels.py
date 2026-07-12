# conformance_levels.py — the L0–L6 ecosystem conformance ladder (RFC-0700 §8,
# ecosystem RFC pack 2026-07-12). Phase-0 "truth repair": report the HONEST current
# level and, crucially, distinguish an achieved level from its evidence NAMESPACE
# (test_only vs production). A level is achieved only when every level below it is.
#
# This is distinct from the older `lerobot-compatibility-levels` contract report (which
# grades individual rungs of the LeRobot policy contract). This ladder grades the whole
# Apple/CoreAI deployment path, per the ecosystem RFCs. Pure Python; no torch/lerobot.

from __future__ import annotations

from dataclasses import dataclass

# ordered ladder; each entry is (id, short title, meaning).
CONFORMANCE_LADDER = (
    ("L0", "Metadata", "Artifact can be inspected"),
    ("L1", "Protocol", "Runner action-profile handshake"),
    ("L2", "Factory", "Official LeRobot factory/plugin loads"),
    ("L3", "Official Eval", "Real official CLI completes the controlled matrix"),
    ("L4", "Real Core AI", "Real Swift Runner executes a real .aimodel"),
    ("L5", "Device Certified", "Signed, scoped device/artifact/runtime certificate"),
    ("L6", "Robot Task Evidence",
     "Guarded physical-run evidence (never equivalent to safety certification)"),
)
_ORDER = [lid for lid, _t, _m in CONFORMANCE_LADDER]

# the signal that MUST be true for each level to be achievable (in ladder order).
_LEVEL_SIGNAL = {
    "L0": "metadata_inspectable",
    "L1": "protocol_handshake",
    "L2": "official_factory_loads",
    "L3": "official_cli_matrix",
    "L4": "real_swift_runner_executes_real_aimodel",
    "L5": "production_signed_certificate",
    "L6": "guarded_robot_task_evidence",
}


@dataclass(frozen=True)
class ConformanceAssessment:
    level: str                     # highest achieved level id, or "" if none
    namespace: str                 # "production" | "test_only"
    achieved: tuple                # achieved level ids, in order
    not_achieved: tuple            # not-yet-achieved level ids, in order

    def to_dict(self) -> dict:
        return {"level": self.level, "namespace": self.namespace,
                "achieved": list(self.achieved), "not_achieved": list(self.not_achieved),
                "ladder": [{"id": i, "title": t, "meaning": m}
                           for i, t, m in CONFORMANCE_LADDER]}


def assess_conformance_level(signals: dict, *, namespace: str = "test_only",
                             ) -> ConformanceAssessment:
    """Highest level whose signal — and every lower level's signal — is true. The ladder
    is monotonic: a gap stops the climb (you cannot be L5 without L4). ``namespace``
    records whether the supporting evidence is production or test_only; an L4+ claim in
    ``test_only`` is NOT a real device certification."""
    achieved: list[str] = []
    for lid in _ORDER:
        if signals.get(_LEVEL_SIGNAL[lid]):
            achieved.append(lid)
        else:
            break
    not_achieved = [lid for lid in _ORDER if lid not in achieved]
    return ConformanceAssessment(
        level=achieved[-1] if achieved else "",
        namespace=namespace if namespace in ("production", "test_only") else "test_only",
        achieved=tuple(achieved), not_achieved=tuple(not_achieved))


def current_conformance() -> ConformanceAssessment:
    """The repository's HONEST current conformance (RFC-0700 §1/§8, 2026-07-12):

    - L0–L3 achieved: metadata inspection, Runner protocol handshake, official LeRobot
      factory/plugin loading, and the real official `lerobot-eval` five-case matrix
      (v1.3.27.3) all exist and pass in CI;
    - L4+ NOT achieved: there is no real Swift Runner executing a real `.aimodel`, no
      production-signed device certificate, and no guarded robot-task evidence;
    - namespace is **test_only**: the L3 matrix runs against a protocol-compatible STUB
      Runner and the executor receipt is unsigned — a production high claim requires an
      executor-signed receipt under a pinned release key (still pending).
    """
    return assess_conformance_level({
        "metadata_inspectable": True,
        "protocol_handshake": True,
        "official_factory_loads": True,
        "official_cli_matrix": True,
        "real_swift_runner_executes_real_aimodel": False,
        "production_signed_certificate": False,
        "guarded_robot_task_evidence": False,
    }, namespace="test_only")
