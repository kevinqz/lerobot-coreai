# policy_execution_contract.py — PolicyExecutionContract v1 (v1.3.26.2).
#
# FeatureContract describes what the TENSORS mean; this describes the inference
# PROGRAM. Without it you can have universality of shapes without universality of
# execution — a single-pass ACT, an iterative Diffusion sampler and a VLA flow model
# have very different execution semantics. Backend-neutral (describes the policy, not
# the runtime provider). Pure Python + JSON; no lerobot/torch.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rollout_evidence_schema import canonical_json_sha256

POLICY_EXECUTION_CONTRACT_SCHEMA_VERSION = "lerobot-coreai.policy-execution-contract.v1"

# the policy execution families LeRobot spans (imitation single-pass, diffusion/flow
# iterative samplers, VLA language+flow, recurrent, world models).
EXECUTION_KINDS = ("single_pass", "iterative_sampler", "vla_flow", "recurrent",
                   "world_model")
GRAPH_ROLES = ("encoder", "vision_encoder", "language_encoder", "tokenizer",
               "denoise_step", "flow_step", "action_head", "decoder",
               "transition_model", "value_head", "recurrent_cell")
STATE_LIFETIMES = ("none", "per_episode", "persistent")
QUEUE_ORIENTATIONS = ("time_major_batched", "none")
CANCELLATION_MODES = ("none", "supported")


class PolicyExecutionContractError(ValueError):
    """Raised when a PolicyExecutionContract is malformed or self-inconsistent."""


_NE_STR = {"type": "string", "minLength": 1}
POLICY_EXECUTION_CONTRACT_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "contract_id", "execution_kind", "graph",
                 "host_side_operations", "rng", "cache", "action_queue", "deadline",
                 "claims"],
    "properties": {
        "schema_version": {"const": POLICY_EXECUTION_CONTRACT_SCHEMA_VERSION},
        "contract_id": _NE_STR,
        "policy_family": {"type": ["string", "null"]},
        "execution_kind": {"enum": list(EXECUTION_KINDS)},
        "graph": {
            "type": "object", "additionalProperties": False,
            "required": ["nodes", "edges"],
            "properties": {
                "nodes": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "role"],
                    "properties": {"name": _NE_STR,
                                   "role": {"enum": list(GRAPH_ROLES)}}}},
                "edges": {"type": "array", "items": {
                    "type": "array", "items": _NE_STR, "minItems": 2, "maxItems": 2}}}},
        "host_side_operations": {"type": "array", "items": {"type": "string"}},
        "tokenizer": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["identity"],
            "properties": {"identity": _NE_STR,
                           "sha256": {"type": ["string", "null"]}}}]},
        "sampling": {"anyOf": [{"type": "null"}, {
            "type": "object", "additionalProperties": False,
            "required": ["algorithm", "steps", "scheduler"],
            "properties": {"algorithm": _NE_STR,
                           "steps": {"type": "integer", "minimum": 1},
                           "scheduler": _NE_STR}}]},
        "rng": {
            "type": "object", "additionalProperties": False,
            "required": ["algorithm", "seed_semantics"],
            "properties": {"algorithm": _NE_STR,
                           "seed_semantics": {"enum": ["per_episode", "per_call",
                                                       "fixed", "none"]}}},
        "cache": {
            "type": "object", "additionalProperties": False,
            "required": ["state_lifetime", "reset_semantics"],
            "properties": {"state_lifetime": {"enum": list(STATE_LIFETIMES)},
                           "reset_semantics": _NE_STR}},
        "action_queue": {
            "type": "object", "additionalProperties": False,
            "required": ["orientation", "horizon", "commit_semantics"],
            "properties": {"orientation": {"enum": list(QUEUE_ORIENTATIONS)},
                           "horizon": {"type": "integer", "minimum": 1},
                           "commit_semantics": _NE_STR}},
        "deadline": {
            "type": "object", "additionalProperties": False,
            "required": ["has_deadline", "cancellation"],
            "properties": {"has_deadline": {"type": "boolean"},
                           "cancellation": {"enum": list(CANCELLATION_MODES)}}},
        "claims": {
            "type": "object", "additionalProperties": False,
            "required": ["policy_execution_contract_verified", "proves_task_success",
                         "proves_physical_safety"],
            "properties": {
                "policy_execution_contract_verified": {"type": "boolean"},
                "proves_task_success": {"const": False},
                "proves_physical_safety": {"const": False}}},
    },
}


def _acyclic(nodes: list, edges: list) -> bool:
    names = {n["name"] for n in nodes}
    if any(a not in names or b not in names for a, b in edges):
        return False
    adj: dict[str, list[str]] = {n["name"]: [] for n in nodes}
    for a, b in edges:
        adj[a].append(b)
    color: dict[str, int] = {}

    def visit(u: str) -> bool:
        color[u] = 1
        for v in adj[u]:
            if color.get(v) == 1 or (color.get(v) != 2 and not visit(v)):
                return False
        color[u] = 2
        return True
    return all(color.get(n["name"]) == 2 or visit(n["name"]) for n in nodes)


def validate_policy_execution_contract(contract: dict) -> list[str]:
    """Structural + semantic validation, fail-closed. Returns errors (empty = ok)."""
    import jsonschema
    try:
        jsonschema.validate(contract, POLICY_EXECUTION_CONTRACT_SCHEMA)
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]
    errs: list[str] = []
    kind = contract["execution_kind"]
    sampling = contract.get("sampling")
    tokenizer = contract.get("tokenizer")
    nodes, edges = contract["graph"]["nodes"], contract["graph"]["edges"]
    names = [n["name"] for n in nodes]
    if len(names) != len(set(names)):
        errs.append("duplicate graph node name")
    if not _acyclic(nodes, edges):
        errs.append("graph is not a DAG (cycle or dangling edge)")
    # execution-kind semantics.
    if kind == "iterative_sampler":
        if not sampling:
            errs.append("iterative_sampler requires a sampling block")
        elif sampling["steps"] < 1:
            errs.append("iterative_sampler requires steps >= 1")
        if not any(n["role"] in ("denoise_step", "flow_step") for n in nodes):
            errs.append("iterative_sampler requires a denoise_step/flow_step node")
    if kind == "single_pass" and sampling is not None and sampling["steps"] > 1:
        errs.append("single_pass must not declare multi-step sampling")
    if kind == "vla_flow" and not tokenizer:
        errs.append("vla_flow requires a tokenizer")
    if kind == "recurrent" and contract["cache"]["state_lifetime"] == "none":
        errs.append("recurrent policy requires a non-none cache state_lifetime")
    return errs


@dataclass
class PolicyExecutionContract:
    contract_id: str
    execution_kind: str
    graph_nodes: list = field(default_factory=list)
    graph_edges: list = field(default_factory=list)
    host_side_operations: list = field(default_factory=list)
    tokenizer: dict | None = None
    sampling: dict | None = None
    rng: dict = field(default_factory=lambda: {"algorithm": "none",
                                               "seed_semantics": "none"})
    cache: dict = field(default_factory=lambda: {"state_lifetime": "none",
                                                 "reset_semantics": "clear_queue"})
    action_queue: dict = field(default_factory=lambda: {
        "orientation": "time_major_batched", "horizon": 1,
        "commit_semantics": "atomic_queue_commit"})
    deadline: dict = field(default_factory=lambda: {"has_deadline": False,
                                                    "cancellation": "none"})
    policy_family: str | None = None

    def to_dict(self) -> dict:
        return {
            "schema_version": POLICY_EXECUTION_CONTRACT_SCHEMA_VERSION,
            "contract_id": self.contract_id, "policy_family": self.policy_family,
            "execution_kind": self.execution_kind,
            "graph": {"nodes": self.graph_nodes, "edges": self.graph_edges},
            "host_side_operations": self.host_side_operations,
            "tokenizer": self.tokenizer, "sampling": self.sampling, "rng": self.rng,
            "cache": self.cache, "action_queue": self.action_queue,
            "deadline": self.deadline,
            "claims": {"policy_execution_contract_verified": False,
                       "proves_task_success": False, "proves_physical_safety": False},
        }

    def sha256(self) -> str:
        d = self.to_dict()
        d.pop("claims", None)
        return canonical_json_sha256(d)

    def validated(self) -> dict:
        d = self.to_dict()
        errs = validate_policy_execution_contract(d)
        if errs:
            raise PolicyExecutionContractError("; ".join(errs))
        d["claims"]["policy_execution_contract_verified"] = True
        return d
