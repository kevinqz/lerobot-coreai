# test_conformance_levels.py — the L0–L6 ladder (RFC-0700 §8) + the Phase-0 truth-repair
# documentation-claim checks (RFC-0800 §3/§14/§20). Pure base package.

from pathlib import Path

from lerobot_coreai.conformance_levels import (
    CONFORMANCE_LADDER, assess_conformance_level, current_conformance,
)

_REPO = Path(__file__).resolve().parents[1]


def test_current_is_l3_test_only():
    a = current_conformance()
    assert a.level == "L3"
    assert a.namespace == "test_only"
    assert a.achieved == ("L0", "L1", "L2", "L3")
    assert a.not_achieved == ("L4", "L5", "L6")


def test_ladder_is_monotonic_a_gap_stops_the_climb():
    # L4/L5 signals true but L3 false → the climb stops at L2 (no skipping).
    a = assess_conformance_level({
        "metadata_inspectable": True, "protocol_handshake": True,
        "official_factory_loads": True, "official_cli_matrix": False,
        "real_swift_runner_executes_real_aimodel": True,
        "production_signed_certificate": True})
    assert a.level == "L2"
    assert "L3" in a.not_achieved and "L4" in a.not_achieved


def test_no_production_high_level_in_current_state():
    # L4+ must NOT be claimed, and the namespace must not be production, until a real
    # Swift Runner + pinned-key signing exist.
    a = current_conformance()
    assert not ({"L4", "L5", "L6"} & set(a.achieved))
    assert a.namespace != "production"


def _text(*rel):
    return "\n".join((_REPO / r).read_text(encoding="utf-8") for r in rel).lower()


def test_readme_states_the_honest_current_level():
    # RFC-0800 §14: docs state current maturity; must match the code truth.
    a = current_conformance()
    readme = (_REPO / "README.md").read_text(encoding="utf-8")
    assert a.level in readme                       # "L3" appears
    assert "test-only" in readme.lower() or "test_only" in readme.lower()


def test_docs_do_not_overclaim_unachieved_levels():
    # RFC-0800 §3/§20 doc-claim check: the docs must not assert a not-yet-achieved level
    # (or production certification) as done.
    blob = _text("README.md", "docs/conformance-levels-l0-l6.md",
                 "docs/official-lerobot-plugin.md")
    forbidden = [
        "l4 achieved", "l5 achieved", "l6 achieved",
        "l4 (real core ai): achieved", "production certified",
        "device certified: achieved", "production-signed certificate issued",
        "real swift runner executes a real .aimodel: achieved",
    ]
    hits = [p for p in forbidden if p in blob]
    assert not hits, f"documentation overclaims: {hits}"


def test_ladder_shape_is_L0_through_L6():
    assert [lid for lid, _t, _m in CONFORMANCE_LADDER] == \
        ["L0", "L1", "L2", "L3", "L4", "L5", "L6"]
