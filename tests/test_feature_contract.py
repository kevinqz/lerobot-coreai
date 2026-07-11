# test_feature_contract.py — FeatureContract v1 (v1.3.24): schema, validation, diff,
# reports, and the canonical ProcessorStageContract builder. Exhaustive negatives.

import jsonschema
import pytest

from lerobot_coreai.feature_contract import (
    FEATURE_CONTRACT_SCHEMA_VERSION, FeatureContract, FeatureSpec,
    NormalizationContract, ValueDomain, feature_contract_from_dict, make_feature_id,
)
from lerobot_coreai.feature_contract_diff import diff_feature_contracts
from lerobot_coreai.feature_contract_reports import (
    build_validation_report, verify_validation_report,
)
from lerobot_coreai.feature_contract_validation import (
    validate_contract_structure, validate_payload_against_feature_contract,
)

RUNNER_IN = "coreai_runner_input.v1"
RUNNER_OUT = "coreai_runner_output.v1"


def _state_spec(stage=RUNNER_IN, owner="policy_preprocessor"):
    return FeatureSpec(
        feature_id=make_feature_id("observation", "observation.state", stage),
        key="observation.state", role="observation", modality="vector", stage=stage,
        required=True, dtype="float32", shape=("S",), axes=("state",), layout=None,
        value_domain=ValueDomain(finite=True), units="rad",
        names=("j1", "j2", "j3", "j4", "j5", "j6", "grip"),
        normalization=NormalizationContract("normalized", "mean_std", owner))


def _image_spec(stage=RUNNER_IN):
    return FeatureSpec(
        feature_id=make_feature_id("observation", "observation.images.front", stage),
        key="observation.images.front", role="observation", modality="image",
        stage=stage, required=True, dtype="float32", shape=("C", "IH", "IW"),
        axes=("channel", "height", "width"), layout="CHW",
        value_domain=ValueDomain(finite=True, minimum=0.0, maximum=1.0,
                                 closed_interval=True),
        normalization=NormalizationContract("normalized", "scale_0_1",
                                            "policy_preprocessor"))


def _action_spec():
    return FeatureSpec(
        feature_id=make_feature_id("action", "action", RUNNER_OUT),
        key="action", role="action", modality="vector", stage=RUNNER_OUT,
        required=True, dtype="float32", shape=("H", "A"), axes=("horizon", "action"),
        layout=None, value_domain=ValueDomain(finite=True),
        names=None, normalization=NormalizationContract("normalized", None,
                                                        "coreai_model"))


def _contract(obs=None, actions=None):
    return FeatureContract(
        contract_id="evo1-so100-coreai", robot_type="so100", policy_path="k/E",
        observations=tuple(obs if obs is not None else (_state_spec(), _image_spec())),
        actions=tuple(actions if actions is not None else (_action_spec(),)))


_SYM = {"S": 7, "C": 3, "IH": 8, "IW": 8, "H": 3, "A": 7}


def _obs_payload():
    return {"observation.state": [0.0] * 7,
            "observation.images.front": [[[0.5] * 8 for _ in range(8)] for _ in range(3)]}


# --- schema ---

def test_contract_matches_schema():
    from importlib.resources import files
    import json
    schema = json.loads(files("lerobot_coreai.schemas").joinpath(
        "feature-contract-v1.schema.json").read_text())
    jsonschema.validate(_contract().to_dict(), schema)


def test_roundtrip_and_hash_stable():
    c = _contract()
    assert feature_contract_from_dict(c.to_dict()).sha256() == c.sha256()
    assert c.sha256().startswith("sha256:")


# --- structural ---

def test_structure_valid():
    assert validate_contract_structure(_contract()) == []


def test_structure_unknown_stage_fails():
    bad = FeatureSpec(**{**_state_spec().__dict__, "stage": "made_up_stage",
                         "feature_id": "observation:observation.state@made_up_stage"})
    assert validate_contract_structure(_contract(obs=(bad,))) != []


def test_structure_names_order_mismatch_fails():
    # names count is checked against a CONCRETE component dim.
    bad = FeatureSpec(**{**_state_spec().__dict__, "shape": (7,), "axes": ("state",),
                         "names": ("only", "three", "names")})
    errs = validate_contract_structure(_contract(obs=(bad,)))
    assert any("names" in e for e in errs)


def test_structure_double_normalization_owner_fails():
    a = _state_spec(stage=RUNNER_IN, owner="policy_preprocessor")
    b = FeatureSpec(**{**_state_spec(stage="lerobot_policy_preprocessor_output.v1",
                                     owner="env_preprocessor").__dict__})
    errs = validate_contract_structure(_contract(obs=(a, b)))
    assert any("double normalization" in e for e in errs)


def test_structure_undeclared_symbol_fails():
    bad = FeatureSpec(**{**_state_spec().__dict__, "shape": ("Z",), "axes": ("state",)})
    assert any("symbol" in e for e in validate_contract_structure(_contract(obs=(bad,))))


def test_structure_horizon_not_allowed_on_env_action():
    bad = FeatureSpec(**{**_action_spec().__dict__, "stage": "environment_action.v1",
                         "feature_id": "action:action@environment_action.v1"})
    assert any("horizon" in e for e in validate_contract_structure(_contract(actions=(bad,))))


# --- payload validation ---

def test_payload_valid():
    r = validate_payload_against_feature_contract(
        _obs_payload(), _contract(), stage=RUNNER_IN, symbols=_SYM)
    assert r.ok and len(r.validated_feature_ids) == 2


def test_missing_required_fails():
    p = _obs_payload(); del p["observation.state"]
    assert not validate_payload_against_feature_contract(
        p, _contract(), stage=RUNNER_IN, symbols=_SYM).ok


def test_unexpected_feature_closed_fails():
    p = _obs_payload(); p["observation.secret"] = [1.0]
    assert not validate_payload_against_feature_contract(
        p, _contract(), stage=RUNNER_IN, symbols=_SYM).ok


def test_shape_mismatch_fails():
    p = _obs_payload(); p["observation.state"] = [0.0] * 6      # S=7 expected
    r = validate_payload_against_feature_contract(
        p, _contract(), stage=RUNNER_IN, symbols=_SYM)
    assert not r.ok and any("shape" in f for f in r.failures)


def test_nonfinite_fails():
    p = _obs_payload(); p["observation.state"] = [float("inf")] + [0.0] * 6
    assert not validate_payload_against_feature_contract(
        p, _contract(), stage=RUNNER_IN, symbols=_SYM).ok


def test_image_range_violation_fails():
    p = _obs_payload()
    p["observation.images.front"] = [[[2.0] * 8 for _ in range(8)] for _ in range(3)]
    r = validate_payload_against_feature_contract(
        p, _contract(), stage=RUNNER_IN, symbols=_SYM)
    assert not r.ok and any("maximum" in f for f in r.failures)


def test_ragged_payload_fails():
    p = _obs_payload(); p["observation.state"] = [0.0, [1.0], 2.0]
    assert not validate_payload_against_feature_contract(
        p, _contract(), stage=RUNNER_IN, symbols=_SYM).ok


def test_symbol_resolution_failure_fails():
    r = validate_payload_against_feature_contract(
        _obs_payload(), _contract(), stage=RUNNER_IN, symbols={"C": 3})  # no S/IH/IW
    assert not r.ok and any("symbol" in f for f in r.failures)


# --- diff ---

def test_diff_dtype_change_is_breaking():
    cand = _contract(obs=(FeatureSpec(**{**_state_spec().__dict__, "dtype": "float64"}),
                          _image_spec()))
    d = diff_feature_contracts(_contract(), cand)
    assert d.is_breaking


def test_diff_new_optional_feature_is_non_breaking():
    extra = FeatureSpec(**{**_state_spec().__dict__, "required": False,
                           "key": "observation.velocity",
                           "feature_id": "observation:observation.velocity@" + RUNNER_IN})
    cand = _contract(obs=(_state_spec(), _image_spec(), extra))
    d = diff_feature_contracts(_contract(), cand)
    assert not d.is_breaking and d.non_breaking


def test_diff_action_names_change_is_breaking():
    a = FeatureSpec(**{**_action_spec().__dict__, "names": ("x", "y")})
    d = diff_feature_contracts(_contract(), _contract(actions=(a,)))
    assert d.is_breaking


# --- reports ---

def test_report_verified_true_on_clean_validation():
    c = _contract()
    r = validate_payload_against_feature_contract(
        _obs_payload(), c, stage=RUNNER_IN, symbols=_SYM)
    report = build_validation_report(c, [r])
    assert report["claims"]["feature_contract_verified"] is True
    ok, errs = verify_validation_report(report, c)
    assert ok, errs


def test_report_hash_tamper_detected():
    c = _contract()
    r = validate_payload_against_feature_contract(
        _obs_payload(), c, stage=RUNNER_IN, symbols=_SYM)
    report = build_validation_report(c, [r])
    report["feature_contract_sha256"] = "sha256:" + "0" * 64
    ok, errs = verify_validation_report(report, c)
    assert not ok


def test_report_not_verified_on_failure():
    c = _contract()
    p = _obs_payload(); del p["observation.state"]
    r = validate_payload_against_feature_contract(p, c, stage=RUNNER_IN, symbols=_SYM)
    report = build_validation_report(c, [r])
    assert report["claims"]["feature_contract_verified"] is False


# --- Phase 0: canonical ProcessorStageContract builder ---

def test_processor_stage_contract_maps_legacy():
    from lerobot_coreai.stages import (
        PROCESSOR_STAGE_CONTRACT_SCHEMA, build_processor_stage_contract,
        processor_stage_contract_sha256,
    )
    c = build_processor_stage_contract(expects="raw_lerobot_observation",
                                       returns="postprocessed_action")
    jsonschema.validate(c, PROCESSOR_STAGE_CONTRACT_SCHEMA)
    assert c["observation"]["target"] == "coreai_runner_input.v1"
    assert c["action"]["target"] == "environment_action.v1"
    assert processor_stage_contract_sha256(c).startswith("sha256:")


def test_processor_stage_contract_unknown_legacy_fails():
    from lerobot_coreai.stages import build_processor_stage_contract
    with pytest.raises(ValueError):
        build_processor_stage_contract(expects="telepathy", returns="postprocessed_action")
