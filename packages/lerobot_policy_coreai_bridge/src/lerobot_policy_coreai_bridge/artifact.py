# artifact.py — canonical, publishable CoreAI-bridge plugin artifact (v1.3.6/1.3.7).
#
# v1.3.6 built a factory-loadable artifact. v1.3.7 HARDENS it before batching
# multiplies it: a strict typed inventory, exact checksum coverage, path-traversal
# and symlink guards, structured secret scanning, real JSON-schema validation,
# version binding, an honest source-provenance reference, and a verification report
# that separates INTEGRITY (unsigned checksum consistency) from AUTHENTICITY
# (cryptographic signature — still false here; signing lands later).
#
#   plugin-artifact/
#   ├── config.json                        (PreTrainedConfig for coreai_bridge)
#   ├── policy_preprocessor.json           (real PolicyProcessorPipeline)
#   ├── policy_postprocessor.json          (real PolicyProcessorPipeline)
#   ├── lerobot-coreai.json                (the CoreAI manifest)
#   ├── plugin_artifact_manifest.json      (index + versions + claims + provenance)
#   ├── plugin_artifact_inventory.json     (typed inventory + artifact_root_sha256)
#   ├── checksums.json                     (must equal the inventory content set)
#   └── README.md
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
    PLUGIN_ARTIFACT_INVENTORY_SCHEMA,
    PLUGIN_ARTIFACT_MANIFEST_SCHEMA,
    PLUGIN_ARTIFACT_VERIFICATION_REPORT_SCHEMA,
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

MANIFEST_FILENAME = "lerobot-coreai.json"
CONFIG_FILENAME = "config.json"
PLUGIN_MANIFEST_FILENAME = "plugin_artifact_manifest.json"
INVENTORY_FILENAME = "plugin_artifact_inventory.json"
CHECKSUMS_FILENAME = "checksums.json"
README_FILENAME = "README.md"
VERIFICATION_REPORT_FILENAME = "plugin_artifact_verification_report.json"

# Content files (covered by the inventory + checksums). The inventory and
# checksums themselves are integrity descriptors and are NOT content files;
# the verification report is generated after verification and is excluded too.
_CONTENT_ROLES = {
    CONFIG_FILENAME: "policy_config",
    PREPROCESSOR_FILENAME: "policy_preprocessor",
    POSTPROCESSOR_FILENAME: "policy_postprocessor",
    MANIFEST_FILENAME: "coreai_manifest",
    PLUGIN_MANIFEST_FILENAME: "plugin_manifest",
    README_FILENAME: "readme",
}
_INTEGRITY_FILES = {INVENTORY_FILENAME, CHECKSUMS_FILENAME}
_NON_CONTENT = _INTEGRITY_FILES | {VERIFICATION_REPORT_FILENAME}

_FORBIDDEN_TRUE = ("official_eval_certified", "upstream_native", "supports_training",
                   "proves_task_success", "proves_physical_safety")

# Structured secret scan: sensitive JSON key substrings + credential-in-URL.
_SECRET_KEYS = ("token", "secret", "password", "api_key", "apikey", "authorization",
                "bearer")
_CREDENTIAL_URL_RE = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class ArtifactError(RuntimeError):
    """Raised when building or verifying a canonical plugin artifact fails."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _artifact_root_sha256(entries: list[dict]) -> str:
    """A single root digest over sorted (path, sha256) pairs."""
    canon = json.dumps(sorted((e["path"], e["sha256"]) for e in entries),
                       separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canon.encode()).hexdigest()


def _find_secret(obj: Any, path: str = "") -> str | None:
    """Return a human path to the first secret found, or None. Public URLs pass."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if isinstance(v, str) and v and any(s in kl for s in _SECRET_KEYS):
                return f"{path}.{k}"
            found = _find_secret(v, f"{path}.{k}")
            if found:
                return found
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found = _find_secret(v, f"{path}[{i}]")
            if found:
                return found
    elif isinstance(obj, str):
        if _CREDENTIAL_URL_RE.search(obj):
            return f"{path} (credential-in-URL)"
    return None


# MARK: - Version binding (pure, unit-testable)

def check_version_compatibility(artifact_versions: dict, installed: dict) -> tuple[bool, str]:
    """Compare artifact-recorded versions against installed ones (v1.3.7).

    Rules: core/plugin lockstep inside the artifact; installed core/plugin share
    the artifact's major and are not OLDER than the artifact; if a LeRobot version
    is recorded, installed LeRobot must share major.minor.
    """
    a_core = artifact_versions.get("lerobot_coreai")
    a_plugin = artifact_versions.get("lerobot_policy_coreai_bridge")
    if a_core != a_plugin:
        return False, f"artifact core/plugin not lockstep ({a_core} != {a_plugin})"
    try:
        for name, key in (("core", "lerobot_coreai"),
                          ("plugin", "lerobot_policy_coreai_bridge")):
            av = Version(artifact_versions[key])
            iv = Version(installed[key])
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
    external_revision: str | None = None,
    external_sha256: str | None = None,  # accepted but re-derived from the manifest
) -> dict[str, Any]:
    """Build a canonical, hardened plugin artifact from a CoreAI artifact."""
    manifest = load_manifest(coreai_artifact, revision=revision)
    require_coreai_processor_ownership(manifest)  # exact v2 semantics

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from lerobot_coreai.action_contract import parse_action_contract_from_manifest
    contract = parse_action_contract_from_manifest(manifest)

    # 1. config.json — coreai_artifact="" (root); only the env-var NAME; cpu device.
    cfg = CoreAIBridgeConfig(
        coreai_artifact="", runner_url_env=runner_url_env,
        minimum_runner_protocol=minimum_runner_protocol,
        expected_action_dim=contract.action_dim,
        expected_action_horizon=(contract.horizon
                                 if contract.representation == "chunk" else None),
        expected_robot_type=getattr(manifest, "robot_type", None),
        device="cpu")
    cfg.save_pretrained(str(out))

    # 2. processors (real PolicyProcessorPipeline; ownership already required).
    pre, post = build_coreai_bridge_processors()
    pre.save_pretrained(str(out), config_filename=PREPROCESSOR_FILENAME)
    post.save_pretrained(str(out), config_filename=POSTPROCESSOR_FILENAME)

    # 3. lerobot-coreai.json (embedded manifest).
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(getattr(manifest, "raw", None) or manifest, indent=2))
    manifest_sha = _sha256_file(out / MANIFEST_FILENAME)
    if external:
        if not external_revision:
            raise ArtifactError(
                "external source references require external_revision (a moving "
                "branch reference is refused); sha256 is derived from the manifest.")
        if external_sha256 and external_sha256 != manifest_sha:
            raise ArtifactError(
                f"external_sha256 {external_sha256} != embedded manifest sha "
                f"{manifest_sha}.")

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
        "files": {
            "config": CONFIG_FILENAME, "preprocessor": PREPROCESSOR_FILENAME,
            "postprocessor": POSTPROCESSOR_FILENAME, "coreai_manifest": MANIFEST_FILENAME,
            "inventory": INVENTORY_FILENAME,
        },
        "versions": {
            "lerobot_coreai": CORE_VERSION,
            "lerobot_policy_coreai_bridge": PLUGIN_VERSION,
            "lerobot": lerobot_v,
        },
        "runner": {"runner_url_env": runner_url_env,
                   "minimum_runner_protocol": minimum_runner_protocol},
        "action_contract": contract.to_dict(),
        # Honest provenance: the RUNTIME manifest is always embedded; an external
        # reference is provenance only (where the source came from).
        "source_coreai_artifact_reference": {
            "mode": "external" if external else "embedded",
            "manifest_sha256": manifest_sha,
            "repo": coreai_artifact if external else None,
            "revision": external_revision if external else None,
        },
        "claims": {
            "official_plugin_factory_compatible": None,  # promoted only by E2E evidence
            "official_eval_certified": False, "upstream_native": False,
            "supports_training": False, "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }
    (out / PLUGIN_MANIFEST_FILENAME).write_text(json.dumps(plugin_manifest, indent=2))
    (out / README_FILENAME).write_text(_render_readme(plugin_manifest))

    # 5. Refuse secrets in config + plugin manifest before sealing.
    for name in (CONFIG_FILENAME, PLUGIN_MANIFEST_FILENAME):
        data = json.loads((out / name).read_text())
        if name == CONFIG_FILENAME and data.get("coreai_artifact"):
            raise ArtifactError(f"{name}: coreai_artifact must be empty (no local path).")
        hit = _find_secret(data)
        if hit:
            raise ArtifactError(f"{name}: refusing to persist a secret at {hit}.")

    # 6. Typed inventory over the content files, then checksums.json (same set).
    entries = []
    for name, role in _CONTENT_ROLES.items():
        p = out / name
        entries.append({"path": name, "role": role, "sha256": _sha256_file(p),
                        "size_bytes": p.stat().st_size})
    entries.sort(key=lambda e: e["path"])
    inventory = {"schema_version": INVENTORY_SCHEMA_VERSION, "files": entries,
                 "artifact_root_sha256": _artifact_root_sha256(entries)}
    jsonschema.validate(inventory, PLUGIN_ARTIFACT_INVENTORY_SCHEMA)
    (out / INVENTORY_FILENAME).write_text(json.dumps(inventory, indent=2))
    (out / CHECKSUMS_FILENAME).write_text(
        json.dumps({e["path"]: e["sha256"] for e in entries}, indent=2))

    jsonschema.validate(plugin_manifest, PLUGIN_ARTIFACT_MANIFEST_SCHEMA)
    return plugin_manifest


def _render_readme(pm: dict[str, Any]) -> str:
    return (
        "# CoreAI Bridge — LeRobot Plugin Artifact\n\n"
        f"policy_type: `{pm['policy_type']}`  ·  schema: `{pm['schema_version']}`\n\n"
        "Load with the official LeRobot factory (see docs/official-lerobot-plugin.md).\n\n"
        f"The runner URL is read from `${pm['runner']['runner_url_env']}` at runtime "
        "and is never stored here.\n\n"
        "Verify integrity with `lerobot-coreai verify-lerobot-plugin-artifact`.\n"
        "This artifact proves **factory/protocol** compatibility and **checksum "
        "integrity** only — NOT cryptographic authenticity, official lerobot-eval, "
        "task success, or physical safety.\n")


# MARK: - Verify

@dataclass
class VerifyResult:
    ok: bool
    checks: dict[str, str] = field(default_factory=dict)
    claims: dict[str, bool] = field(default_factory=dict)
    plugin_manifest: dict[str, Any] | None = None
    artifact_root_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks, "claims": self.claims}


def _safe_name(name: str) -> bool:
    return (name == Path(name).name and not Path(name).is_absolute()
            and ".." not in name.split("/") and name not in ("", ".", ".."))


def verify_plugin_artifact(artifact_dir: str, *, deep: bool = True,
                           write_report: bool = False) -> VerifyResult:
    """Verify a canonical plugin artifact, fail-closed.

    Separates INTEGRITY (structure + inventory + checksums + no-tamper, unsigned)
    from AUTHENTICITY (cryptographic signature — always False here). ``deep=True``
    adds lerobot-dependent checks (config parse, processor reload, ownership,
    version binding vs installed).
    """
    out = Path(artifact_dir)
    checks: dict[str, str] = {}

    def ok(name: str, cond: bool, reason: str = "") -> bool:
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond

    integrity_ok = True

    # --- structural presence ---
    required = list(_CONTENT_ROLES) + [INVENTORY_FILENAME, CHECKSUMS_FILENAME]
    for fn in required:
        if not ok(f"present:{fn}", (out / fn).exists(), "missing"):
            integrity_ok = False
    if not integrity_ok:
        return _finish(out, False, checks, None, None, write_report)

    # --- parse inventory + manifest, schema-validate ---
    try:
        inventory = json.loads((out / INVENTORY_FILENAME).read_text())
        jsonschema.validate(inventory, PLUGIN_ARTIFACT_INVENTORY_SCHEMA)
        ok("inventory_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("inventory_schema", False, str(exc))
        return _finish(out, False, checks, None, None, write_report)
    try:
        pm = json.loads((out / PLUGIN_MANIFEST_FILENAME).read_text())
        jsonschema.validate(pm, PLUGIN_ARTIFACT_MANIFEST_SCHEMA)
        ok("plugin_manifest_schema", True)
    except Exception as exc:  # noqa: BLE001
        ok("plugin_manifest_schema", False, str(exc))
        return _finish(out, False, checks, None, None, write_report)

    inv_paths = [e["path"] for e in inventory["files"]]

    # --- path safety (before opening anything by inventory path) ---
    path_ok = all(_safe_name(p) for p in inv_paths)
    integrity_ok &= ok("inventory_paths_safe", path_ok, "unsafe path in inventory")

    # --- no symlinks / no undeclared files ---
    declared = set(inv_paths) | _NON_CONTENT
    escape_ok, undeclared_ok = True, True
    for p in out.iterdir():
        if p.is_symlink():
            escape_ok = False
        if p.name not in declared:
            undeclared_ok = False
            checks[f"undeclared:{p.name}"] = "failed: not in inventory"
    integrity_ok &= ok("no_symlinks", escape_ok, "symlink present")
    integrity_ok &= ok("no_undeclared_files", undeclared_ok, "undeclared file present")

    # --- every declared content file present + digest + size match ---
    for e in inventory["files"]:
        if not _safe_name(e["path"]):
            continue
        fp = out / e["path"]
        if not fp.exists():
            integrity_ok &= ok(f"inv:{e['path']}", False, "missing")
            continue
        digest = _sha256_file(fp)
        size = fp.stat().st_size
        good = (digest == e["sha256"] and size == e["size_bytes"]
                and _SHA256_RE.match(e["sha256"]))
        integrity_ok &= ok(f"inv:{e['path']}", bool(good),
                           f"{digest}/{size} != {e['sha256']}/{e['size_bytes']}")

    # --- artifact_root recompute ---
    recomputed = _artifact_root_sha256(inventory["files"])
    integrity_ok &= ok("artifact_root_sha256", recomputed == inventory["artifact_root_sha256"],
                       f"{recomputed} != {inventory['artifact_root_sha256']}")

    # --- checksums.json == inventory content set, exactly ---
    try:
        recorded = json.loads((out / CHECKSUMS_FILENAME).read_text())
    except Exception as exc:  # noqa: BLE001
        recorded = None
        integrity_ok &= ok("checksums_parse", False, str(exc))
    if recorded is not None:
        inv_map = {e["path"]: e["sha256"] for e in inventory["files"]}
        exact = set(recorded) == set(inv_map)
        integrity_ok &= ok("checksums_exact_coverage", exact,
                           f"{sorted(recorded)} != {sorted(inv_map)}")
        digests_ok = all(recorded.get(k) == v for k, v in inv_map.items())
        integrity_ok &= ok("checksums_match_inventory", digests_ok, "digest mismatch")
        malformed = [k for k, v in recorded.items() if not _SHA256_RE.match(str(v))]
        integrity_ok &= ok("checksums_well_formed", not malformed,
                           f"malformed digests: {malformed}")

    # --- source reference: manifest_sha matches embedded file; external pinned ---
    ref = pm["source_coreai_artifact_reference"]
    man_sha = _sha256_file(out / MANIFEST_FILENAME)
    integrity_ok &= ok("source_manifest_sha", ref["manifest_sha256"] == man_sha,
                       f"{ref['manifest_sha256']} != {man_sha}")
    if ref["mode"] == "external":
        integrity_ok &= ok("external_reference_pinned",
                           bool(ref.get("revision")) and bool(_SHA256_RE.match(ref["manifest_sha256"])),
                           "external reference missing revision / valid sha256")

    # --- no forbidden claims; no secrets ---
    claims_block = pm.get("claims", {})
    integrity_ok &= ok("no_forbidden_claims",
                       all(claims_block.get(k) is not True for k in _FORBIDDEN_TRUE),
                       "a forbidden claim is asserted true")
    secret_free = True
    for name in (CONFIG_FILENAME, PLUGIN_MANIFEST_FILENAME):
        hit = _find_secret(json.loads((out / name).read_text()))
        if hit:
            secret_free = False
            checks[f"secret:{name}"] = f"failed: {hit}"
    integrity_ok &= ok("no_secrets", secret_free, "secret detected")

    processor_ok = False
    if deep:
        integrity_ok, processor_ok = _verify_deep(out, pm, checks, ok, integrity_ok)

    authenticity_ok = False  # no signature integration yet (v1.3.9). Honest.
    return _finish(out, integrity_ok, checks, pm,
                   inventory.get("artifact_root_sha256"), write_report,
                   authenticity=authenticity_ok, processor_contract=processor_ok)


def _verify_deep(out, pm, checks, ok, integrity_ok):
    """lerobot-dependent checks: version binding, config parse, processors, ownership."""
    # version binding.
    ok_ver, reason = check_version_compatibility(pm["versions"], _installed_versions(True))
    integrity_ok &= ok("version_compatibility", ok_ver, reason)
    # config parses.
    try:
        import lerobot_policy_coreai_bridge  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(str(out))
        integrity_ok &= ok("config_parses", cfg.type == POLICY_TYPE, f"type={cfg.type!r}")
        integrity_ok &= ok("config_no_local_path", not cfg.coreai_artifact,
                           "coreai_artifact not empty")
    except Exception as exc:  # noqa: BLE001
        integrity_ok &= ok("config_parses", False, str(exc))
    # processors reload.
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
    # processor contract (exact v2 ownership semantics).
    processor_ok = False
    try:
        m = json.loads((out / MANIFEST_FILENAME).read_text())
        processor_ok = manifest_declares_coreai_ownership(m)
        integrity_ok &= ok("processor_contract", processor_ok,
                           "manifest lacks exact CoreAI processor ownership")
    except Exception as exc:  # noqa: BLE001
        integrity_ok &= ok("processor_contract", False, str(exc))
    return integrity_ok, processor_ok


def _finish(out, integrity_ok, checks, pm, root_sha, write_report,
            *, authenticity=False, processor_contract=False):
    claims = {
        "integrity_verified": bool(integrity_ok),
        "authenticity_verified": bool(authenticity),
        "processor_contract_verified": bool(processor_contract),
        "factory_b1_certified": False,      # promoted only by the signed cert (v1.3.9)
        "official_eval_certified": False,
        "proves_physical_safety": False,
    }
    result = VerifyResult(ok=bool(integrity_ok), checks=checks, claims=claims,
                          plugin_manifest=pm, artifact_root_sha256=root_sha)
    if write_report:
        report = {"schema_version": REPORT_SCHEMA_VERSION,
                  "artifact_root_sha256": root_sha, "checks": checks, "claims": claims}
        jsonschema.validate(report, PLUGIN_ARTIFACT_VERIFICATION_REPORT_SCHEMA)
        (Path(out) / VERIFICATION_REPORT_FILENAME).write_text(json.dumps(report, indent=2))
        (Path(out) / "plugin_artifact_verification_report.md").write_text(
            _render_report_md(report))
    return result


def _render_report_md(report: dict) -> str:
    lines = ["# Plugin Artifact Verification Report", "",
             f"artifact_root_sha256: `{report.get('artifact_root_sha256')}`", "",
             "## Claims", ""]
    for k, v in report["claims"].items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Checks", ""]
    for k, v in report["checks"].items():
        lines.append(f"- {'✓' if v == 'passed' else '✗'} {k}: {v}")
    lines += ["", "_Integrity = unsigned checksum consistency. Authenticity requires "
              "a trusted signature (not yet issued)._"]
    return "\n".join(lines) + "\n"
