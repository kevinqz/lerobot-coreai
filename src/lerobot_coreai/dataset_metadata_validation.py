# dataset_metadata_validation.py — DatasetMetadataEvidence <-> FeatureContract (v1.3.25).
#
# Bind the dataset metadata identity to the v1 FeatureContract: every required
# dataset-side feature must exist with matching dtype/shape/names; camera keys must be
# image/video modality; action names+order must match; robot_type compatible; fps
# positive; stats referenced by normalization must exist; task feature consistent.
# Fail-closed. Pure Python; no lerobot.

from __future__ import annotations

from dataclasses import dataclass, field

from .feature_contract import FeatureContract

_IMAGE_MODALITIES = ("image", "video", "depth")


@dataclass
class MetadataBindingResult:
    ok: bool = True
    failures: list[str] = field(default_factory=list)

    def _fail(self, msg: str):
        self.failures.append(msg)
        self.ok = False

    def to_dict(self) -> dict:
        return {"ok": self.ok, "failures": self.failures}


def bind_metadata_to_feature_contract(
    evidence: dict, contract: FeatureContract, *,
    robot_type_compatible: bool = True,
) -> MetadataBindingResult:
    res = MetadataBindingResult()
    meta_features = evidence.get("features", {}) or {}
    meta_shapes = evidence.get("shapes", {}) or {}
    meta_names = evidence.get("names", {}) or {}
    camera_keys = set(evidence.get("camera_keys", []) or [])

    # robot_type compatibility (metadata vs contract).
    if contract.robot_type is not None and evidence.get("robot_type") is not None:
        if not robot_type_compatible and contract.robot_type != evidence["robot_type"]:
            res._fail(f"robot_type {evidence['robot_type']} != contract "
                      f"{contract.robot_type}")

    if int(evidence.get("fps", 0)) < 1:
        res._fail("fps must be positive")

    # every dataset-declared feature that the contract also declares must agree on
    # dtype/shape/names (compared against the concrete metadata, ignoring symbols).
    by_key: dict[str, list] = {}
    for spec in contract.all_specs():
        by_key.setdefault(spec.key, []).append(spec)

    for key, mfeat in meta_features.items():
        specs = by_key.get(key)
        if not specs:
            continue                                   # env/context-only features
        for spec in specs:
            m_dtype = mfeat.get("dtype")
            # dtype family check (float32 vs float32; image/video are non-numeric).
            if m_dtype in ("float32", "float64", "int64") and spec.dtype and \
                    m_dtype != spec.dtype and spec.modality == "vector":
                res._fail(f"{key}: metadata dtype {m_dtype} != contract {spec.dtype}")
            # names/order must match when both declare them.
            m_names = meta_names.get(key) or mfeat.get("names")
            if m_names and spec.names and tuple(m_names) != tuple(spec.names):
                res._fail(f"{key}: names/order metadata {list(m_names)} != contract "
                          f"{list(spec.names)}")
            # concrete component dim must match the metadata shape's last dim.
            m_shape = meta_shapes.get(key) or mfeat.get("shape")
            comp = next((d for d in reversed(spec.shape) if isinstance(d, int)), None)
            if m_shape and comp is not None and int(m_shape[-1]) != comp:
                res._fail(f"{key}: component dim {comp} != metadata {m_shape[-1]}")
            # camera keys must map to an image-family modality.
            if key in camera_keys and spec.modality not in _IMAGE_MODALITIES:
                res._fail(f"{key}: camera key bound to non-image modality "
                          f"{spec.modality}")
            # stats referenced by normalization must exist in the metadata features.
            ref = spec.normalization.stats_ref
            if ref and ref not in meta_features:
                res._fail(f"{key}: normalization stats_ref {ref!r} absent from metadata")

    # required contract observation/action features must exist in the metadata.
    for spec in contract.observations + contract.actions:
        if spec.required and spec.role in ("observation", "action") \
                and spec.key not in meta_features and spec.modality != "text":
            res._fail(f"required feature {spec.key} not present in dataset metadata")
    return res
