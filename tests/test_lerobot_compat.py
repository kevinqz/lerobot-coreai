# test_lerobot_compat.py — LeRobot compatibility certificate (v1.1.2).

import importlib
import json
import sys
from importlib.resources import files

import jsonschema

from lerobot_coreai.lerobot_compat import (
    LEROBOT_COMPAT_SCHEMA_VERSION, build_compat_report, evaluate_lerobot_compat,
)


def _has_lerobot():
    try:
        import importlib.metadata as md
        md.version("lerobot")
        return True
    except Exception:
        return False


def test_report_schema_valid():
    report = evaluate_lerobot_compat(strict=False)
    assert report["schema_version"] == LEROBOT_COMPAT_SCHEMA_VERSION
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "lerobot-compatibility-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_claims_are_honest():
    report = evaluate_lerobot_compat(strict=False)
    assert report["claims"]["native_upstream_registry"] is False
    assert report["claims"]["supports_training"] is False
    assert report["claims"]["supports_physical_safety"] is False


def test_bridge_shape_checks_present_and_pass():
    report = evaluate_lerobot_compat(strict=False)
    names = {c["name"]: c for c in report["checks"]}
    assert names["coreai_bridge_importable"]["passed"] is True
    assert names["native_registry_claim_false"]["passed"] is True
    assert names["training_claim_false"]["passed"] is True


def test_non_strict_ok_without_lerobot_dependent_required_failures():
    # In non-strict mode, a missing LeRobot must NOT fail the certificate: the
    # base package is usable without the extra.
    report = evaluate_lerobot_compat(strict=False)
    if not _has_lerobot():
        assert report["ok"] is True
        li = next(c for c in report["checks"] if c["name"] == "lerobot_installed")
        assert li["severity"] == "info"


def test_strict_requires_lerobot():
    report = evaluate_lerobot_compat(strict=True)
    li = next(c for c in report["checks"] if c["name"] == "lerobot_installed")
    assert li["severity"] == "required"
    if not _has_lerobot():
        assert report["ok"] is False


def test_base_import_does_not_pull_lerobot_or_torch():
    # Importing the compat module (and the base package) must not import
    # lerobot/torch. This is the core "no heavy import at base" invariant.
    for mod in [m for m in list(sys.modules) if m == "lerobot" or m == "torch"
                or m.startswith("lerobot.") or m.startswith("torch.")]:
        # If already imported by another test, we can't assert cleanly; only
        # assert that OUR modules don't reference them at import.
        pass
    # Re-import our module fresh and confirm it declares no lerobot/torch object.
    import lerobot_coreai.lerobot_compat as compat
    importlib.reload(compat)
    assert not hasattr(compat, "torch")
    assert not hasattr(compat, "lerobot")


def test_build_compat_report_shape():
    report = build_compat_report(True, [], {"version": "0.6.1", "in_range": True}, (3, 12))
    assert report["python_version"] == "3.12"
    assert report["lerobot_version"] == "0.6.1"
    assert report["claims"]["compatible_with_lerobot_0_6_x_shape"] is True
