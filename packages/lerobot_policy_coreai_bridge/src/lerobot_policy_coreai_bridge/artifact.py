# artifact.py — canonical, publishable CoreAI-bridge plugin artifact (v1.3.6–1.3.8).
#
# v1.3.6 built a factory-loadable artifact; v1.3.7 hardened its STRUCTURE (typed
# inventory, exact checksums, path/symlink guards, integrity≠authenticity). v1.3.8
# closes its SEMANTICS: every embedded contract is schema-validated (processor,
# action, claims), the files are cross-bound to a single source of truth, secrets
# are scanned across ALL declared JSON, provenance carries an immutable resolved
# commit sha, the artifact root digest binds role+size, and verification reports
# are written OUTSIDE the sealed artifact (verify is idempotent).
#
#   plugin-artifact/
#   ├── config.json / policy_preprocessor.json / policy_postprocessor.json
#   ├── lerobot-coreai.json / plugin_artifact_manifest.json
#   ├── plugin_artifact_inventory.json / checksums.json / README.md
#
# Never persists a runner URL, token, secret, or machine-local absolute path.
# No hardware, no egress.

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from packaging.version import InvalidVersion, Version

from lerobot_coreai.manifest import load_manifest

from . import __version__ as PLUGIN_VERSION
from .artifact_schemas import (
    ACTION_CONTRACT_SCHEMA,
    ARTIFACT_ROLES,
    CLAIMS_SCHEMA,
    PLUGIN_ARTIFACT_INVENTORY_SCHEMA,
    PLUGIN_ARTIFACT_MANIFEST_SCHEMA,
    PLUGIN_ARTIFACT_VERIFICATION_REPORT_SCHEMA,
    PROCESSOR_CONTRACT_SCHEMA,
)
from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig
from .processor_coreai_bridge import (
    POSTPROCESSOR_FILENAME,
    PREPROCESSOR_FILENAME,
    build_coreai_bridge_processors,
    manifest_declares_coreai_ownership,
    require_coreai_processor_ownership,
)

ARTIFACT_SCHEMA_VERSION = "lerobot-coreai.plugin_artifact.v1"
INVENTORY_SCHEMA_VERSION = "lerobot-coreai.plugin_inventory.v1"
REPORT_SCHEMA_VERSION = "lerobot-coreai.plugin_artifact_verification.v1"
ROOT_ALGORITHM = "canonical-json-sha256.v1"

MANIFEST_FILENAME = "lerobot-coreai.json"
CONFIG_FILENAME = "config.json"
PLUGIN_MANIFEST_FILENAME = "plugin_artifact_manifest.json"
INVENTORY_FILENAME = "plugin_artifact_inventory.json"
CHECKSUMS_FILENAME = "checksums.json"
README_FILENAME = "README.md"

# role -> filename (exactly one file per role).
_ROLE_FILE = {
    "policy_config": CONFIG_FILENAME,
    "policy_preprocessor": PREPROCESSOR_FILENAME,
    "policy_postprocessor": POSTPROCESSOR_FILENAME,
    "coreai_manifest": MANIFEST_FILENAME,
    "plugin_manifest": PLUGIN_MANIFEST_FILENAME,
    "readme": README_FILENAME,
}
_CONTENT_FILES = set(_ROLE_FILE.values())
_INTEGRITY_FILES = {INVENTORY_FILENAME, CHECKSUMS_FILENAME}
# JSON content files scanned for secrets (README is text -> URL-only scan).
_JSON_CONTENT = (CONFIG_FILENAME, PLUGIN_MANIFEST_FILENAME, MANIFEST_FILENAME,
                 PREPROCESSOR_FILENAME, POSTPROCESSOR_FILENAME)

_SECRET_KEYS = ("token", "secret", "password", "api_key", "apikey", "authorization",
                "bearer")
_CREDENTIAL_URL_RE = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ArtifactError(RuntimeError):
    """Raised when building or verifying a canonical plugin artifact fails."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_obj(obj: Any) -> str:
    """Canonical sha256 of a JSON-serializable object (sorted keys)."""
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def _artifact_root_sha256(entries: list[dict]) -> str:
    """Root digest over sorted full entries (path, role, sha256, size) + algorithm."""
    canon = json.dumps(
        {"algorithm": ROOT_ALGORITHM, "schema_version": INVENTORY_SCHEMA_VERSION,
         "files": sorted(({"path": e["path"], "role": e["role"], "sha256": e["sha256"],
                           "size_bytes": e["size_bytes"]} for e in entries),
                         key=lambda e: e["path"])},
        separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def _find_secret(obj: Any, path: str = "") -> str | None:
    """First secret path, or None. Sensitive key + ANY non-empty value fails."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if any(s in kl for s in _SECRET_KEYS) and v not in (None, "", {}, [], ()):
                return f"{path}.{k}"
            found = _find_secret(v, f"{path}.{k}")
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found = _find_secret(v, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(obj, str) and _CREDENTIAL_URL_RE.search(obj):
        return f"{path} (credential-in-URL)"
    return None


# MARK: - Version binding (pure)

def check_version_compatibility(artifact_versions: dict, installed: dict) -> tuple[bool, str]:
    a_core = artifact_versions.get("lerobot_coreai")
    a_plugin = artifact_versions.get("lerobot_policy_coreai_bridge")
    if a_core != a_plugin:
        return False, f"artifact core/plugin not lockstep ({a_core} != {a_plugin})"
    try:
        for name, key in (("core", "lerobot_coreai"),
                          ("plugin", "lerobot_policy_coreai_bridge")):
            av, iv = Version(artifact_versions[key]), Version(installed[key])
            if av.major != iv.major:
                return False, f"{name} major mismatch (artifact {av} vs installed {iv})"
            if iv < av:
                return False, f"installed {name} {iv} is older than artifact {av}"
        a_lr, i_lr = artifact_versions.get("lerobot"), installed.get("lerobot")
        if a_lr and i_lr:
            avl, ivl = Version(a_lr), Version(i_lr)
            if (avl.major, avl.minor) != (ivl.major, ivl.minor):
                return False, f"LeRobot minor mismatch (artifact {avl} vs installed {ivl})"
    except (InvalidVersion, KeyError) as exc:
        return False, f"unparseable version: {exc}"
    return True, "ok"


def _installed_versions(deep: bool) -> dict:
    from lerobot_coreai import __version__ as core_v
    out = {"lerobot_coreai": core_v, "lerobot_policy_coreai_bridge": PLUGIN_VERSION,
           "lerobot": None}
    if deep:
        try:
            import lerobot
            out["lerobot"] = getattr(lerobot, "__version__", None)
        except Exception:  # noqa: BLE001
            out["lerobot"] = None
    return out


# MARK: - Build

def build_plugin_artifact(
    coreai_artifact: str,
    output_dir: str,
    *,
    runner_url_env: str = "COREAI_RUNNER_URL",
    minimum_runner_protocol: str = "coreai-runner.v2",
    revision: str = "main",
    external: bool = False,
    requested_ref: str | None = None,
    resolved_commit_sha: str | None = None,
) -> dict[str, Any]:
    """Build a canonical, hardened, semantically-closed plugin artifact."""
    manifest = load_manifest(coreai_artifact, revision=revision)
    require_coreai_processor_ownership(manifest)  # exact v2 semantics

    # Validate the embedded processor contract against its schema (v1.3.8).
    proc_contract = _manifest_processor_contract(getattr(manifest, "raw", None) or {})
    jsonschema.validate(proc_contract, PROCESSOR_CONTRACT_SCHEMA)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from lerobot_coreai.action_contract import (
        parse_action_contract_from_manifest, parse_batch_contract_from_manifest)
    contract = parse_action_contract_from_manifest(manifest)
    batch_contract = parse_batch_contract_from_manifest(manifest)

    # 1. config.json — coreai_artifact="" (root); env-var NAME only; cpu device.
    cfg = CoreAIBridgeConfig(
        coreai_artifact="", runner_url_env=runner_url_env,
        minimum_runner_protocol=minimum_runner_protocol,
        expected_action_dim=contract.action_dim,
        expected_action_horizon=(contract.horizon
                                 if contract.representation == "chunk" else None),
        expected_robot_type=getattr(manifest, "robot_type", None),
        device="cpu")
    cfg.save_pretrained(str(out))

    # 2. processors (real PolicyProcessorPipeline).
    pre, post = build_coreai_bridge_processors()
    pre.save_pretrained(str(out), config_filename=PREPROCESSOR_FILENAME)
    post.save_pretrained(str(out), config_filename=POSTPROCESSOR_FILENAME)

    # 3. lerobot-coreai.json (embedded manifest).
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(getattr(manifest, "raw", None) or manifest, indent=2))
    manifest_sha = _sha256_file(out / MANIFEST_FILENAME)

    # Provenance: immutable resolved commit for external release references.
    if external:
        if not resolved_commit_sha or not _COMMIT_RE.match(resolved_commit_sha):
            raise ArtifactError(
                "external references require resolved_commit_sha (40-hex); a mutable "
                f"ref like {requested_ref!r} alone is refused.")
    source_ref = {
        "mode": "external" if external else "embedded",
        "embedded_manifest_sha256": manifest_sha,
        "source_repo": coreai_artifact if external else None,
        "requested_ref": requested_ref if external else None,
        "resolved_commit_sha": resolved_commit_sha if external else None,
    }

    # 4. plugin_artifact_manifest.json
    from lerobot_coreai import __version__ as CORE_VERSION
    try:
        import lerobot
        lerobot_v = getattr(lerobot, "__version__", None)
    except Exception:  # noqa: BLE001
        lerobot_v = None
    plugin_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "policy_type": POLICY_TYPE,
        "files": {"config": CONFIG_FILENAME, "preprocessor": PREPROCESSOR_FILENAME,
                  "postprocessor": POSTPROCESSOR_FILENAME,
                  "coreai_manifest": MANIFEST_FILENAME, "inventory": INVENTORY_FILENAME},
        "versions": {"lerobot_coreai": CORE_VERSION,
                     "lerobot_policy_coreai_bridge": PLUGIN_VERSION, "lerobot": lerobot_v},
        "runner": {"runner_url_env": runner_url_env,
                   "minimum_runner_protocol": minimum_runner_protocol},
        "action_contract": contract.to_dict(),
        "batch_contract": batch_contract.to_dict(),
        "batch_contract_sha256": _sha256_obj(batch_contract.to_dict()),
        "processor_stage_contract": {
            "observation_input_stage": batch_contract.observation_stage,
            "action_output_stage": "postprocessed_environment_action.v1",
        },
        "source_coreai_artifact_reference": source_ref,
        "claims": {"official_plugin_factory_compatible": None,
                   "official_eval_certified": False, "upstream_native": False,
                   "supports_training": False, "proves_task_success": False,
                   "proves_physical_safety": False},
    }
    jsonschema.validate(plugin_manifest, PLUGIN_ARTIFACT_MANIFEST_SCHEMA)
    (out / PLUGIN_MANIFEST_FILENAME).write_text(json.dumps(plugin_manifest, indent=2))
    (out / README_FILENAME).write_text(_render_readme(plugin_manifest))

    # 5. Refuse secrets across all declared JSON before sealing.
    if json.loads((out / CONFIG_FILENAME).read_text()).get("coreai_artifact"):
        raise ArtifactError("config.json: coreai_artifact must be empty (no local path).")
    for name in _JSON_CONTENT:
        hit = _find_secret(json.loads((out / name).read_text()))
        if hit:
            raise ArtifactError(f"{name}: refusing to persist a secret at {hit}.")

    # 6. Typed inventory (unique path+role, role enum) + checksums (== inventory).
    entries = []
    for role, name in _ROLE_FILE.items():
        p = out / name
        entries.append({"path": name, "role": role, "sha256": _sha256_file(p),
                        "size_bytes": p.stat().st_size})
    entries.sort(key=lambda e: e["path"])
    inventory = {"schema_version": INVENTORY_SCHEMA_VERSION,
                 "artifact_root_algorithm": ROOT_ALGORITHM, "files": entries,
                 "artifact_root_sha256": _artifact_root_sha256(entries)}
    jsonschema.validate(inventory, PLUGIN_ARTIFACT_INVENTORY_SCHEMA)
    (out / INVENTORY_FILENAME).write_text(json.dumps(inventory, indent=2))
    (out / CHECKSUMS_FILENAME).write_text(
        json.dumps({e["path"]: e["sha256"] for e in entries}, indent=2))
    return plugin_manifest


def _manifest_processor_contract(manifest_raw: dict) -> dict:
    return (manifest_raw.get("contracts", {}) or {}).get("processor", {}) or {}


def _render_readme(pm: dict[str, Any]) -> str:
    return (
        "# CoreAI Bridge — LeRobot Plugin Artifact\n\n"
        f"policy_type: `{pm['policy_type']}`  ·  schema: `{pm['schema_version']}`\n\n"
        "Load with the official LeRobot factory (see docs/official-lerobot-plugin.md).\n\n"
        f"The runner URL is read from `${pm['runner']['runner_url_env']}` at runtime "
        "and is never stored here.\n\n"
        "Verify with `lerobot-coreai verify-lerobot-plugin-artifact` (reports are "
        "written OUTSIDE this sealed artifact).\n"
        "Proves **factory/protocol** compatibility and **checksum integrity** only — "
        "NOT cryptographic authenticity, official lerobot-eval, task success, or "
        "physical safety.\n")


# MARK: - Semantic cross-binding (v1.3.8, lerobot-free)

def verify_artifact_semantics(out: Path) -> dict[str, str]:
    """Cross-bind config/plugin-manifest/coreai-manifest/inventory/processors.

    Returns a name->status map ("passed" | "failed: …" | "not_verified: …").
    Absent/unknowable properties are "not_verified", never silently "passed".
    """
    from lerobot_coreai.action_contract import (
        parse_action_contract_from_manifest, parse_batch_contract_from_manifest)
    from lerobot_coreai.manifest import LeRobotCoreAIManifest
    checks: dict[str, str] = {}

    def c(name, cond, reason=""):
        checks[name] = "passed" if cond else f"failed: {reason}"

    cfg = json.loads((out / CONFIG_FILENAME).read_text())
    pm = json.loads((out / PLUGIN_MANIFEST_FILENAME).read_text())
    coreai = json.loads((out / MANIFEST_FILENAME).read_text())
    inventory = json.loads((out / INVENTORY_FILENAME).read_text())
    # Parse the action contract via the manifest OBJECT (same path as build), so
    # feature-derived shapes are read consistently with plugin_artifact_manifest.
    coreai_obj = LeRobotCoreAIManifest.from_dict(coreai)

    # processor contract schema + exact ownership.
    proc = _manifest_processor_contract(coreai)
    try:
        jsonschema.validate(proc, PROCESSOR_CONTRACT_SCHEMA)
        c("processor_contract_schema", True)
    except Exception as exc:  # noqa: BLE001
        c("processor_contract_schema", False, str(exc))
    c("processor_ownership_exact", manifest_declares_coreai_ownership(coreai),
      "manifest lacks exact CoreAI ownership")

    # action contract equality: plugin manifest vs parsed CoreAI manifest.
    contract = parse_action_contract_from_manifest(coreai_obj).to_dict()
    c("action_contract_equality", pm.get("action_contract") == contract,
      f"{pm.get('action_contract')} != {contract}")

    # batch contract equality + hash (v1.3.11): the plugin manifest must record the
    # SAME batch contract the CoreAI manifest declares, hash-bound.
    batch = parse_batch_contract_from_manifest(coreai_obj).to_dict()
    c("batch_contract_equality", pm.get("batch_contract") == batch,
      f"{pm.get('batch_contract')} != {batch}")
    c("batch_contract_sha256", pm.get("batch_contract_sha256") == _sha256_obj(batch),
      "batch_contract_sha256 mismatch")
    # processor stage: input stage must match the batch contract's observation_stage.
    stage = pm.get("processor_stage_contract", {})
    c("processor_stage_binding",
      stage.get("observation_input_stage") == batch.get("observation_stage"),
      f"{stage.get('observation_input_stage')} != {batch.get('observation_stage')}")

    # config expectations vs manifest.
    c("config_action_dim", cfg.get("expected_action_dim") == contract["action_dim"],
      f"{cfg.get('expected_action_dim')} != {contract['action_dim']}")
    expected_h = contract["horizon"] if contract["representation"] == "chunk" else None
    c("config_action_horizon", cfg.get("expected_action_horizon") == expected_h,
      f"{cfg.get('expected_action_horizon')} != {expected_h}")
    robot_type = (coreai.get("robot", {}) or {}).get("type")
    c("config_robot_type", cfg.get("expected_robot_type") == robot_type,
      f"{cfg.get('expected_robot_type')} != {robot_type}")
    c("protocol_equality",
      pm.get("runner", {}).get("minimum_runner_protocol") == cfg.get("minimum_runner_protocol"),
      "minimum_runner_protocol differs between plugin manifest and config")

    # files/roles ↔ inventory.
    inv_roles = {e["role"]: e["path"] for e in inventory["files"]}
    role_ok = all(inv_roles.get(r) == f for r, f in _ROLE_FILE.items())
    c("inventory_role_file_mapping", role_ok, f"role→file mismatch: {inv_roles}")
    pm_files = set(pm.get("files", {}).values()) - {INVENTORY_FILENAME}
    c("manifest_files_in_inventory", pm_files <= set(inv_roles.values()),
      f"plugin manifest files not all inventoried: {pm_files}")

    # processor step structure vs identity contract (step-empty pipelines).
    for fn, key in ((PREPROCESSOR_FILENAME, "preprocessor"),
                    (POSTPROCESSOR_FILENAME, "postprocessor")):
        try:
            steps = json.loads((out / fn).read_text()).get("steps", None)
            c(f"processor_steps_empty:{key}", steps == [],
              f"{key} is not step-empty (identity contract requires no steps)")
        except Exception as exc:  # noqa: BLE001
            c(f"processor_steps_empty:{key}", False, str(exc))

    # feature semantics: shapes verifiable from the manifest; dtype/names/units/
    # layout are recorded as not_verified (config PolicyFeatures don't carry them).
    checks["feature_dtype"] = "not_verified: config features carry no dtype"
    checks["feature_action_names_order"] = "not_verified: manifest declares no action names"
    checks["feature_image_layout_range"] = "not_verified: no image features declared"
    return checks


# MARK: - Verify

@dataclass
class VerifyResult:
    ok: bool
    checks: dict[str, str] = field(default_factory=dict)
    semantics: dict[str, str] = field(default_factory=dict)
    claims: dict[str, bool] = field(default_factory=dict)
    plugin_manifest: dict[str, Any] | None = None
    artifact_root_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "semantics": self.semantics,
                "claims": self.claims}


def _safe_name(name: str) -> bool:
    return (name == Path(name).name and not Path(name).is_absolute()
            and ".." not in name.split("/") and name not in ("", ".", ".."))


def verify_plugin_artifact(artifact_dir: str, *, deep: bool = True,
                           report_dir: str | None = None) -> VerifyResult:
    """Verify a canonical plugin artifact, fail-closed. Never writes into it.

    Reports (when ``report_dir`` is given) are written OUTSIDE the sealed artifact,
    so verification is idempotent and the artifact root never changes.
    """
    out = Path(artifact_dir)
    checks: dict[str, str] = {}
    integrity_ok = True

    def ok(name: str, cond: bool, reason: str = "") -> bool:
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond

    required = list(_CONTENT_FILES) + [INVENTORY_FILENAME, CHECKSUMS_FILENAME]
    for fn in required:
        if not ok(f"present:{fn}", (out / fn).exists(), "missing"):
            integrity_ok = False
    if not integrity_ok:
        return _finish(out, False, checks, {}, None, None, report_dir)

    try:
        inventory = json.loads((out / INVENTORY_FILENAME).read_text())
        jsonschema.validate(inventory, PLUGIN_ARTIFACT_INVENTORY_SCHEMA)
        ok("inventory_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("inventory_schema", False, str(exc))
        return _finish(out, False, checks, {}, None, None, report_dir)
    try:
        pm = json.loads((out / PLUGIN_MANIFEST_FILENAME).read_text())
        jsonschema.validate(pm, PLUGIN_ARTIFACT_MANIFEST_SCHEMA)
        ok("plugin_manifest_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("plugin_manifest_schema", False, str(exc))
        return _finish(out, False, checks, {}, None, None, report_dir)

    inv_paths = [e["path"] for e in inventory["files"]]
    inv_roles = [e["role"] for e in inventory["files"]]

    # uniqueness (paths + roles) — schema uniqueItems covers whole entries only.
    integrity_ok &= ok("inventory_unique_paths", len(inv_paths) == len(set(inv_paths)),
                       "duplicate path")
    integrity_ok &= ok("inventory_unique_roles", len(inv_roles) == len(set(inv_roles)),
                       "duplicate role")
    integrity_ok &= ok("inventory_paths_safe", all(_safe_name(p) for p in inv_paths),
                       "unsafe path in inventory")

    declared = set(inv_paths) | _INTEGRITY_FILES
    escape_ok, undeclared_ok = True, True
    for p in out.iterdir():
        if p.is_symlink():
            escape_ok = False
        if p.name not in declared:
            undeclared_ok = False
            checks[f"undeclared:{p.name}"] = "failed: not in inventory"
    integrity_ok &= ok("no_symlinks", escape_ok, "symlink present")
    integrity_ok &= ok("no_undeclared_files", undeclared_ok, "undeclared file present")

    for e in inventory["files"]:
        if not _safe_name(e["path"]):
            continue
        fp = out / e["path"]
        if not fp.exists():
            integrity_ok &= ok(f"inv:{e['path']}", False, "missing")
            continue
        digest, size = _sha256_file(fp), fp.stat().st_size
        good = (digest == e["sha256"] and size == e["size_bytes"])
        integrity_ok &= ok(f"inv:{e['path']}", bool(good),
                           f"{digest}/{size} != {e['sha256']}/{e['size_bytes']}")

    integrity_ok &= ok("artifact_root_sha256",
                       _artifact_root_sha256(inventory["files"]) == inventory["artifact_root_sha256"],
                       "root digest mismatch")

    try:
        recorded = json.loads((out / CHECKSUMS_FILENAME).read_text())
        inv_map = {e["path"]: e["sha256"] for e in inventory["files"]}
        integrity_ok &= ok("checksums_exact_coverage", set(recorded) == set(inv_map),
                           f"{sorted(recorded)} != {sorted(inv_map)}")
        integrity_ok &= ok("checksums_match_inventory",
                           all(recorded.get(k) == v for k, v in inv_map.items()),
                           "digest mismatch")
        integrity_ok &= ok("checksums_well_formed",
                           all(_SHA256_RE.match(str(v)) for v in recorded.values()),
                           "malformed digest")
    except Exception as exc:  # noqa: BLE001
        integrity_ok &= ok("checksums_parse", False, str(exc))

    # source provenance.
    ref = pm["source_coreai_artifact_reference"]
    man_sha = _sha256_file(out / MANIFEST_FILENAME)
    integrity_ok &= ok("source_manifest_sha", ref["embedded_manifest_sha256"] == man_sha,
                       "embedded manifest sha mismatch")
    if ref["mode"] == "external":
        integrity_ok &= ok("external_resolved_commit",
                           bool(ref.get("resolved_commit_sha")
                                and _COMMIT_RE.match(ref["resolved_commit_sha"] or "")),
                           "external reference missing/invalid resolved_commit_sha")

    # secrets across ALL declared JSON + README URL scan.
    secret_free = True
    for name in _JSON_CONTENT:
        hit = _find_secret(json.loads((out / name).read_text()))
        if hit:
            secret_free = False
            checks[f"secret:{name}"] = f"failed: {hit}"
    if _CREDENTIAL_URL_RE.search((out / README_FILENAME).read_text()):
        secret_free = False
        checks["secret:README.md"] = "failed: credential-in-URL"
    integrity_ok &= ok("no_secrets", secret_free, "secret detected")

    # --- semantics (lerobot-free cross-binding) ---
    semantics = verify_artifact_semantics(out)
    # Consistency = nothing failed; completeness = everything passed (v1.3.9).
    consistency_ok = all(not v.startswith("failed") for v in semantics.values())
    completeness_ok = all(v == "passed" for v in semantics.values())
    integrity_ok &= ok("semantic_consistency", consistency_ok,
                       "a semantic cross-binding check failed")

    processor_ok = manifest_declares_coreai_ownership(
        json.loads((out / MANIFEST_FILENAME).read_text()))

    if deep:
        integrity_ok = _verify_deep(out, pm, checks, ok, integrity_ok)

    return _finish(out, integrity_ok, checks, semantics, pm,
                   inventory.get("artifact_root_sha256"), report_dir,
                   processor_contract=processor_ok,
                   consistency=consistency_ok, completeness=completeness_ok)


def _verify_deep(out, pm, checks, ok, integrity_ok):
    ok_ver, reason = check_version_compatibility(pm["versions"], _installed_versions(True))
    integrity_ok &= ok("version_compatibility", ok_ver, reason)
    try:
        import lerobot_policy_coreai_bridge  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(str(out))
        integrity_ok &= ok("config_parses", cfg.type == POLICY_TYPE, f"type={cfg.type!r}")
        integrity_ok &= ok("config_no_local_path", not cfg.coreai_artifact,
                           "coreai_artifact not empty")
    except Exception as exc:  # noqa: BLE001
        integrity_ok &= ok("config_parses", False, str(exc))
    try:
        from lerobot.processor import (
            PolicyProcessorPipeline, batch_to_transition, policy_action_to_transition,
            transition_to_batch, transition_to_policy_action)
        PolicyProcessorPipeline.from_pretrained(
            str(out), config_filename=PREPROCESSOR_FILENAME,
            to_transition=batch_to_transition, to_output=transition_to_batch)
        PolicyProcessorPipeline.from_pretrained(
            str(out), config_filename=POSTPROCESSOR_FILENAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action)
        integrity_ok &= ok("processors_reload", True)
    except Exception as exc:  # noqa: BLE001
        integrity_ok &= ok("processors_reload", False, str(exc))
    return integrity_ok


def _finish(out, integrity_ok, checks, semantics, pm, root_sha, report_dir,
            *, authenticity=False, processor_contract=False,
            consistency=False, completeness=False):
    claims = {
        "integrity_verified": bool(integrity_ok),
        "authenticity_verified": bool(authenticity),
        "processor_contract_verified": bool(processor_contract),
        # Honest split (v1.3.9): consistency = nothing failed; completeness = all
        # passed. not_verified aspects keep completeness false without failing.
        "semantic_consistency_verified": bool(consistency),
        "semantic_completeness_verified": bool(completeness),
        "factory_b1_certified": False,      # promoted only by the signed cert (v1.3.10)
        "official_eval_certified": False,
        "proves_physical_safety": False,
    }
    result = VerifyResult(ok=bool(integrity_ok), checks=checks, semantics=semantics,
                          claims=claims, plugin_manifest=pm, artifact_root_sha256=root_sha)
    if report_dir:
        rd = Path(report_dir)
        rd.mkdir(parents=True, exist_ok=True)
        report = {"schema_version": REPORT_SCHEMA_VERSION, "artifact_root_sha256": root_sha,
                  "checks": checks, "semantics": semantics, "claims": claims}
        jsonschema.validate(report, PLUGIN_ARTIFACT_VERIFICATION_REPORT_SCHEMA)
        (rd / "plugin_artifact_verification_report.json").write_text(
            json.dumps(report, indent=2))
        (rd / "plugin_artifact_verification_report.md").write_text(_render_report_md(report))
    return result


def _render_report_md(report: dict) -> str:
    lines = ["# Plugin Artifact Verification Report", "",
             f"artifact_root_sha256: `{report.get('artifact_root_sha256')}`", "",
             "## Claims", ""]
    for k, v in report["claims"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Checks", ""]
    for k, v in {**report["checks"], **report.get("semantics", {})}.items():
        lines.append(f"- {'✓' if v == 'passed' else ('~' if v.startswith('not_verified') else '✗')} {k}: {v}")
    lines += ["", "_Integrity = unsigned checksum + semantic consistency. Authenticity "
              "requires a trusted signature (not yet issued)._"]
    return "\n".join(lines) + "\n"
