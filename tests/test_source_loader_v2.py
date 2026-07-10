# test_source_loader_v2.py — official-API source loader (v1.2.6, hardened v1.2.7).

import json
import types
from importlib.resources import files
from unittest.mock import patch

import jsonschema
import pytest

from lerobot_coreai import source_loader_v2 as sl
from lerobot_coreai.source_loader_v2 import (
    SOURCE_POLICY_LOAD_SCHEMA_VERSION, SourceLoaderError, build_source_load_report,
    load_source_policy,
)


def _cfg():
    # Must be attribute-settable so the loader can bind cfg.pretrained_path.
    return types.SimpleNamespace()


class _Policy:  # a distinct class so policy_class is recorded
    pass


def _patch_all():
    return (
        patch.object(sl, "_load_config", return_value=_cfg()),
        patch.object(sl, "_load_dataset_metadata",
                     return_value=types.SimpleNamespace(stats={})),
        patch.object(sl, "_make_policy", return_value=_Policy()),
        patch.object(sl, "_make_processors", return_value=(object(), object())),
    )


def test_load_binds_pretrained_path_and_uses_official_api():
    p1, p2, p3, p4 = _patch_all()
    with p1 as m1, p2, p3 as m3, p4:
        bundle = load_source_policy("lerobot/diffusion_pusht", "lerobot/pusht",
                                    policy_revision="abc123")
    assert bundle.pretrained_path_bound is True
    assert str(bundle.config.pretrained_path).endswith("diffusion_pusht")
    assert bundle.policy_class == "_Policy"
    m1.assert_called_once()
    m3.assert_called_once()


def test_load_failure_reports_stage_policy():
    with patch.object(sl, "_load_config", return_value=_cfg()), \
         patch.object(sl, "_load_dataset_metadata", return_value=_cfg()), \
         patch.object(sl, "_make_policy", side_effect=RuntimeError("boom")):
        with pytest.raises(SourceLoaderError) as ei:
            load_source_policy("p", "d")
    assert ei.value.stage == "policy"


def test_load_failure_reports_stage_config():
    with patch.object(sl, "_load_config", side_effect=RuntimeError("no cfg")):
        with pytest.raises(SourceLoaderError) as ei:
            load_source_policy("p", "d")
    assert ei.value.stage == "config"


def test_report_ok_proves_weights_bound_and_schema_valid():
    p1, p2, p3, p4 = _patch_all()
    with p1, p2, p3, p4:
        bundle = load_source_policy("p", "d")
    report = build_source_load_report("p", "d", bundle=bundle)
    assert report["schema_version"] == SOURCE_POLICY_LOAD_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["weights"]["pretrained_path_bound"] is True
    assert report["claims"]["proves_source_weights_bound"] is True
    assert report["claims"]["proves_training_support"] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "source-policy-load-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_report_failure_records_stage():
    err = SourceLoaderError("processors", "nope")
    report = build_source_load_report("p", "d", bundle=None, error=err)
    assert report["ok"] is False
    assert report["failed_stage"] == "processors"
    assert report["weights"]["pretrained_path_bound"] is False
