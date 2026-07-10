# test_source_loader_v2.py — official-API source loader (v1.2.6).

import json
from importlib.resources import files
from unittest.mock import patch

import jsonschema
import pytest

from lerobot_coreai import source_loader_v2 as sl
from lerobot_coreai.source_loader_v2 import (
    SOURCE_POLICY_LOAD_SCHEMA_VERSION, SourceLoaderError, build_source_load_report,
    load_source_policy,
)


def _patch_all():
    return (
        patch.object(sl, "_load_config", return_value=object()),
        patch.object(sl, "_load_dataset_metadata", return_value=type("M", (), {"stats": {}})()),
        patch.object(sl, "_make_policy", return_value=object()),
        patch.object(sl, "_make_processors", return_value=(object(), object())),
    )


def test_load_uses_official_api():
    p1, p2, p3, p4 = _patch_all()
    with p1 as m1, p2 as m2, p3 as m3, p4 as m4:
        bundle = load_source_policy("lerobot/diffusion_pusht", "lerobot/pusht")
    assert bundle.policy is not None and bundle.preprocessor is not None
    m1.assert_called_once()
    m3.assert_called_once()  # make_policy called with cfg + ds_meta, not a string
    m4.assert_called_once()


def test_load_failure_reports_stage_policy():
    with patch.object(sl, "_load_config", return_value=object()), \
         patch.object(sl, "_load_dataset_metadata", return_value=object()), \
         patch.object(sl, "_make_policy", side_effect=RuntimeError("boom")):
        with pytest.raises(SourceLoaderError) as ei:
            load_source_policy("p", "d")
    assert ei.value.stage == "policy"


def test_load_failure_reports_stage_config():
    with patch.object(sl, "_load_config", side_effect=RuntimeError("no cfg")):
        with pytest.raises(SourceLoaderError) as ei:
            load_source_policy("p", "d")
    assert ei.value.stage == "config"


def test_report_ok_and_honest_and_schema_valid():
    p1, p2, p3, p4 = _patch_all()
    with p1, p2, p3, p4:
        bundle = load_source_policy("p", "d")
    report = build_source_load_report("p", "d", bundle=bundle)
    assert report["schema_version"] == SOURCE_POLICY_LOAD_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["loader"]["used_official_api"] is True
    assert report["loader"]["instantiated_pretrained_base"] is False
    assert report["loader"]["used_string_policy_type"] is False
    assert report["claims"]["proves_training_support"] is False
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "source-policy-load-report.schema.json").read_text())
    jsonschema.validate(report, schema)


def test_report_failure_records_stage():
    err = SourceLoaderError("processors", "nope")
    report = build_source_load_report("p", "d", bundle=None, error=err)
    assert report["ok"] is False
    assert report["failed_stage"] == "processors"
