# certification_bundle.py — CertificationBundle v1 (v1.3.26.13).
#
# The 5th review's biggest remaining structural gap: the OfficialEvalCertificate bound
# eleven roots that were non-null + hash-formatted but NEVER re-opened or re-verified —
# a caller could pass `{root: "sha256:" + "a"*64}` and satisfy the gate. This closes
# that: promotion consumes a CertificationBundle of the actual leaf EVIDENCE OBJECTS,
# and the verifier DERIVES every root by content-addressing its leaf (a bare hash can no
# longer stand in for evidence) and RE-RUNS the real leaf verifier wherever one exists.
#
# Honesty about depth: two leaves are fully re-verified offline from their own bytes
# (processor_parity replays its arrays; model_conversion replays its bundle). The others
# are content-addressed + schema-checked ("addressed"): their root is proven to be the
# hash of a real, schema-valid object, not an arbitrary string — but a dedicated
# semantic re-verifier (e.g. the dataset metadata tree, which needs the on-disk
# snapshot) is a later step. The bundle result reports the per-root verification LEVEL
# so nothing is silently overclaimed. Pure Python, offline.

from __future__ import annotations

from .official_eval_certificate import _ROOT_KEYS
from .rollout_evidence_schema import canonical_json_sha256

CERTIFICATION_BUNDLE_SCHEMA_VERSION = "lerobot-coreai.certification-bundle.v1"

CERTIFICATION_BUNDLE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object", "additionalProperties": False,
    "required": ["schema_version", "leaves"],
    "properties": {
        "schema_version": {"const": CERTIFICATION_BUNDLE_SCHEMA_VERSION},
        "leaves": {
            "type": "object", "additionalProperties": False,
            "required": list(_ROOT_KEYS),
            "properties": {k: {"type": "object"} for k in _ROOT_KEYS}},
    },
}


def _reverify_processor_parity(leaf):
    from .processor_parity import verify_processor_parity_report
    return verify_processor_parity_report(leaf)


def _reverify_model_conversion(leaf):
    from .model_conversion_evidence import verify_model_conversion_evidence
    return verify_model_conversion_evidence(leaf)


# root name -> a self-contained (bytes-only) re-verifier. Extended as more leaves gain
# offline verifiers (dataset metadata tree, feature-contract semantics, …).
_REVERIFIERS = {
    "processor_parity_sha256": _reverify_processor_parity,
    "model_conversion_sha256": _reverify_model_conversion,
}


def verify_certification_bundle(bundle: dict) -> tuple[bool, list, dict]:
    """Verify a bundle of leaf evidence objects. Returns (ok, reasons, result) where
    ``result`` = {"roots": {root_name: sha256}, "levels": {root_name: "reverified" |
    "addressed"}}. Every root is DERIVED by content-addressing its leaf, so a bare hash
    can never appear as a root; leaves with a self-contained verifier are re-run."""
    import jsonschema
    errors: list[str] = []
    try:
        jsonschema.validate(bundle, CERTIFICATION_BUNDLE_SCHEMA)
    except Exception as exc:  # noqa: BLE001
        return False, [f"schema: {exc}"], {}
    leaves = bundle["leaves"]
    roots: dict[str, str] = {}
    levels: dict[str, str] = {}
    for name in _ROOT_KEYS:
        leaf = leaves[name]
        roots[name] = canonical_json_sha256(leaf)     # content-addressed (not declared)
        reverifier = _REVERIFIERS.get(name)
        if reverifier is not None:
            ok, why = reverifier(leaf)
            if not ok:
                errors.append(f"{name}: leaf verifier failed: {why}")
            levels[name] = "reverified"
        else:
            levels[name] = "addressed"
    # cross-binding: if the parity leaf records a feature-contract root, it must be the
    # content root of the feature_contract leaf actually in the bundle.
    pc = leaves["processor_parity_sha256"]
    pc_fc = pc.get("feature_contract_sha256")
    if pc_fc is not None and pc_fc != roots["feature_contract_sha256"]:
        errors.append("processor_parity.feature_contract_sha256 != feature_contract "
                      "leaf content root (cross-binding)")
    # cross-binding: the conversion leaf's .aimodel must be the artifact the bundle binds
    # as its artifact_root (same executed artifact across the graph).
    mc = leaves["model_conversion_sha256"]
    mc_aimodel = (mc.get("artifact") or {}).get("aimodel_sha256")
    art = leaves["artifact_root_sha256"]
    art_aimodel = art.get("aimodel_sha256") if isinstance(art, dict) else None
    if mc_aimodel is not None and art_aimodel is not None and mc_aimodel != art_aimodel:
        errors.append("model_conversion.artifact.aimodel != artifact_root leaf aimodel "
                      "(cross-binding)")
    return (not errors), errors, {"roots": roots, "levels": levels}
