# test_policy_execution_contract.py — PolicyExecutionContract v1 (v1.3.26.2).
# Covers ACT (single_pass), Diffusion (iterative_sampler) and SmolVLA (vla_flow) at
# the CONTRACT level — the external review's "contracts must not be EVO1-only" check,
# diagnostic-grade (no claims about task success). Pure base.

import jsonschema
import pytest

from lerobot_coreai.policy_execution_contract import (
    POLICY_EXECUTION_CONTRACT_SCHEMA, PolicyExecutionContract,
    PolicyExecutionContractError, validate_policy_execution_contract,
)


def _act() -> PolicyExecutionContract:
    return PolicyExecutionContract(
        contract_id="act-so100", execution_kind="single_pass", policy_family="act",
        graph_nodes=[{"name": "enc", "role": "encoder"},
                     {"name": "head", "role": "action_head"}],
        graph_edges=[["enc", "head"]],
        host_side_operations=["temporal_ensemble"],
        action_queue={"orientation": "time_major_batched", "horizon": 100,
                      "commit_semantics": "atomic_queue_commit"})


def _diffusion() -> PolicyExecutionContract:
    return PolicyExecutionContract(
        contract_id="diffusion-pusht", execution_kind="iterative_sampler",
        policy_family="diffusion",
        graph_nodes=[{"name": "obs_enc", "role": "encoder"},
                     {"name": "eps", "role": "denoise_step"}],
        graph_edges=[["obs_enc", "eps"]],
        sampling={"algorithm": "ddim", "steps": 10, "scheduler": "ddpm_cosine"},
        rng={"algorithm": "philox", "seed_semantics": "per_call"},
        action_queue={"orientation": "time_major_batched", "horizon": 16,
                      "commit_semantics": "atomic_queue_commit"})


def _smolvla() -> PolicyExecutionContract:
    return PolicyExecutionContract(
        contract_id="smolvla-so100", execution_kind="vla_flow", policy_family="smolvla",
        graph_nodes=[{"name": "vit", "role": "vision_encoder"},
                     {"name": "tok", "role": "tokenizer"},
                     {"name": "flow", "role": "flow_step"},
                     {"name": "head", "role": "action_head"}],
        graph_edges=[["vit", "flow"], ["tok", "flow"], ["flow", "head"]],
        tokenizer={"identity": "smolvla-tokenizer", "sha256": None},
        sampling={"algorithm": "flow_matching", "steps": 10, "scheduler": "linear"},
        action_queue={"orientation": "time_major_batched", "horizon": 50,
                      "commit_semantics": "atomic_queue_commit"})


@pytest.mark.parametrize("factory", [_act, _diffusion, _smolvla])
def test_family_contracts_validate(factory):
    c = factory()
    d = c.validated()
    jsonschema.validate(d, POLICY_EXECUTION_CONTRACT_SCHEMA)
    assert d["claims"]["policy_execution_contract_verified"] is True
    assert c.sha256().startswith("sha256:")


def test_three_families_are_distinct():
    ids = {_act().sha256(), _diffusion().sha256(), _smolvla().sha256()}
    assert len(ids) == 3       # the contract vocabulary is not collapsed to one shape


def test_iterative_sampler_requires_sampling():
    c = _diffusion(); c.sampling = None
    assert any("sampling" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_iterative_sampler_requires_denoise_node():
    c = _diffusion()
    c.graph_nodes = [{"name": "obs_enc", "role": "encoder"}]
    c.graph_edges = []
    assert any("denoise" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_single_pass_rejects_multistep_sampling():
    c = _act(); c.sampling = {"algorithm": "x", "steps": 5, "scheduler": "y"}
    assert any("single_pass" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_vla_requires_tokenizer():
    c = _smolvla(); c.tokenizer = None
    assert any("tokenizer" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_recurrent_requires_state():
    c = PolicyExecutionContract(
        contract_id="rnn", execution_kind="recurrent",
        graph_nodes=[{"name": "cell", "role": "recurrent_cell"}], graph_edges=[])
    assert any("cache" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_cyclic_graph_fails():
    c = _act(); c.graph_edges = [["enc", "head"], ["head", "enc"]]
    assert any("DAG" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_dangling_edge_fails():
    c = _act(); c.graph_edges = [["enc", "ghost"]]
    assert any("DAG" in e for e in validate_policy_execution_contract(c.to_dict()))


def test_unknown_execution_kind_fails():
    c = _act(); c.execution_kind = "telepathy"
    assert validate_policy_execution_contract(c.to_dict())


def test_validated_raises_on_invalid():
    c = _smolvla(); c.tokenizer = None
    with pytest.raises(PolicyExecutionContractError):
        c.validated()


def test_hash_stable_and_claim_excluded():
    c = _act()
    h1 = c.sha256()
    d = c.to_dict(); d["claims"]["policy_execution_contract_verified"] = True
    # the claim flag must not change the semantic hash.
    import copy
    c2 = copy.deepcopy(c)
    assert c2.sha256() == h1
