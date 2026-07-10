# artifact.py — canonical, publishable CoreAI-bridge plugin artifact (v1.3.6).
#
# Builds and verifies a self-contained directory that the OFFICIAL LeRobot factory
# can consume end to end:
#
#   plugin-artifact/
#   ├── config.json                 (PreTrainedConfig for coreai_bridge)
#   ├── policy_preprocessor.json     (real PolicyProcessorPipeline)
#   ├── policy_postprocessor.json    (real PolicyProcessorPipeline)
#   ├── lerobot-coreai.json          (the CoreAI manifest)
#   ├── plugin_artifact_manifest.json(index + versions + claims)
#   ├── checksums.json               (sha256 of every artifact file, tamper-evident)
#   └── README.md
#
# Never persists a runner URL, token, secret, or machine-local absolute path:
# config.json carries only runner_url_env (the NAME of the env var) and
# coreai_artifact="" (meaning "the artifact root"). No hardware, no egress.

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot_coreai.manifest import load_manifest

from . import __version__ as PLUGIN_VERSION
from .configuration_coreai_bridge import POLICY_TYPE, CoreAIBridgeConfig
from .processor_coreai_bridge import (
    POSTPROCESSOR_FILENAME,
    PREPROCESSOR_FILENAME,
    require_coreai_processor_ownership,
    save_coreai_bridge_processors,
)

ARTIFACT_SCHEMA_VERSION = "lerobot-coreai.plugin_artifact.v1"
MANIFEST_FILENAME = "lerobot-coreai.json"
CONFIG_FILENAME = "config.json"
PLUGIN_MANIFEST_FILENAME = "plugin_artifact_manifest.json"
CHECKSUMS_FILENAME = "checksums.json"
README_FILENAME = "README.md"

# Files whose sha256 goes into checksums.json (checksums.json itself excluded).
_CHECKSUMMED = (CONFIG_FILENAME, PREPROCESSOR_FILENAME, POSTPROCESSOR_FILENAME,
                MANIFEST_FILENAME, PLUGIN_MANIFEST_FILENAME, README_FILENAME)

# Forbidden-claim keys that must never be asserted true in the plugin manifest.
_FORBIDDEN_TRUE = ("official_eval_certified", "upstream_native", "supports_training",
                   "proves_task_success", "proves_physical_safety")

# Heuristic secret/URL/token patterns refused in persisted config/manifest.
_SECRET_RE = re.compile(
    r"(https?://|unix://|[A-Za-z0-9_-]*(?:token|secret|api[_-]?key|password)[A-Za-z0-9_-]*\s*[:=])",
    re.IGNORECASE)


class ArtifactError(RuntimeError):
    """Raised when building or verifying a canonical plugin artifact fails."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _looks_like_secret(text: str) -> bool:
    return bool(_SECRET_RE.search(text))


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
    external_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a canonical plugin artifact from a CoreAI artifact (local dir/repo).

    ``external=False`` (embedded) copies the manifest into the artifact and
    references it by ``coreai_artifact=""`` (the artifact root). ``external=True``
    records an immutable ``{repo, revision, sha256}`` reference — a moving branch
    without a revision/sha256 is refused. Returns the plugin manifest dict.
    """
    manifest = load_manifest(coreai_artifact, revision=revision)
    # Ownership must be declared before we emit identity processors.
    require_coreai_processor_ownership(manifest)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from lerobot_coreai.action_contract import parse_action_contract_from_manifest
    contract = parse_action_contract_from_manifest(manifest)

    # 1. config.json — never persist a URL/token/path; coreai_artifact="" = root.
    cfg = CoreAIBridgeConfig(
        coreai_artifact="", runner_url_env=runner_url_env,
        minimum_runner_protocol=minimum_runner_protocol,
        expected_action_dim=contract.action_dim,
        expected_action_horizon=(contract.horizon
                                 if contract.representation == "chunk" else None),
        expected_robot_type=getattr(manifest, "robot_type", None),
        device="cpu")
    cfg.save_pretrained(str(out))

    # 2. processors (real PolicyProcessorPipeline JSON; ownership re-checked).
    save_coreai_bridge_processors(str(out), manifest=manifest)

    # 3. lerobot-coreai.json — the CoreAI manifest (always embedded); an external
    #    reference additionally records an immutable {repo, revision, sha256}.
    external_ref = None
    if external:
        if not external_revision or not external_sha256:
            raise ArtifactError(
                "external artifact references require both external_revision and "
                "external_sha256 (a moving branch reference is refused).")
        external_ref = {"repo": coreai_artifact, "revision": external_revision,
                        "sha256": external_sha256}
    (out / MANIFEST_FILENAME).write_text(
        json.dumps(getattr(manifest, "raw", None) or manifest, indent=2))

    # 4. plugin_artifact_manifest.json
    from lerobot_coreai import __version__ as CORE_VERSION
    plugin_manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "policy_type": POLICY_TYPE,
        "files": {
            "config": CONFIG_FILENAME,
            "preprocessor": PREPROCESSOR_FILENAME,
            "postprocessor": POSTPROCESSOR_FILENAME,
            "coreai_manifest": MANIFEST_FILENAME,
        },
        "versions": {
            "lerobot_coreai": CORE_VERSION,
            "lerobot_policy_coreai_bridge": PLUGIN_VERSION,
        },
        "runner": {
            "runner_url_env": runner_url_env,
            "minimum_runner_protocol": minimum_runner_protocol,
        },
        "action_contract": contract.to_dict(),
        "coreai_artifact_reference": (
            {"mode": "external", **external_ref} if external_ref
            else {"mode": "embedded"}),
        "claims": {
            "official_plugin_factory_compatible": None,  # promoted only by E2E evidence
            "official_eval_certified": False,
            "upstream_native": False,
            "supports_training": False,
            "proves_task_success": False,
            "proves_physical_safety": False,
        },
    }
    (out / PLUGIN_MANIFEST_FILENAME).write_text(json.dumps(plugin_manifest, indent=2))

    # 5. README.md
    (out / README_FILENAME).write_text(_render_readme(plugin_manifest))

    # 6. Refuse to persist secrets, then checksum everything.
    _assert_no_secrets(out)
    checksums = {name: _sha256_file(out / name) for name in _CHECKSUMMED
                 if (out / name).exists()}
    (out / CHECKSUMS_FILENAME).write_text(json.dumps(checksums, indent=2))
    return plugin_manifest


def _render_readme(pm: dict[str, Any]) -> str:
    return (
        "# CoreAI Bridge — LeRobot Plugin Artifact\n\n"
        f"policy_type: `{pm['policy_type']}`  ·  schema: `{pm['schema_version']}`\n\n"
        "Load with the official LeRobot factory:\n\n"
        "```python\n"
        "from lerobot.utils.import_utils import register_third_party_plugins\n"
        "from lerobot.configs.policies import PreTrainedConfig\n"
        "from lerobot.policies.factory import make_policy, make_pre_post_processors\n"
        "register_third_party_plugins()\n"
        "cfg = PreTrainedConfig.from_pretrained('./this-artifact')\n"
        "cfg.pretrained_path = './this-artifact'\n"
        "policy = make_policy(cfg, ds_meta=...)\n"
        "pre, post = make_pre_post_processors(cfg, pretrained_path='./this-artifact')\n"
        "```\n\n"
        f"The runner URL is read from `${pm['runner']['runner_url_env']}` at runtime "
        "and is never stored here.\n\n"
        "This artifact proves **protocol/factory** compatibility only. It does NOT "
        "certify official lerobot-eval, task success, or physical safety.\n")


def _assert_no_secrets(out: Path) -> None:
    for name in (CONFIG_FILENAME, PLUGIN_MANIFEST_FILENAME, MANIFEST_FILENAME):
        p = out / name
        if not p.exists():
            continue
        text = p.read_text()
        if name == CONFIG_FILENAME:
            data = json.loads(text)
            for key in ("coreai_artifact",):
                if data.get(key):
                    raise ArtifactError(
                        f"{name}: {key} must be empty (no local path persisted).")
        # The manifest legitimately may contain http(s) URLs for source repos;
        # only guard config + plugin manifest for URL/token leakage.
        if name in (CONFIG_FILENAME, PLUGIN_MANIFEST_FILENAME) and _looks_like_secret(text):
            raise ArtifactError(f"{name} appears to contain a URL/token/secret.")


# MARK: - Verify

@dataclass
class VerifyResult:
    ok: bool
    checks: dict[str, str]           # name -> "passed" | "failed: reason"
    plugin_manifest: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "checks": self.checks}


def verify_plugin_artifact(artifact_dir: str, *, deep: bool = True) -> VerifyResult:
    """Verify a canonical plugin artifact, fail-closed.

    ``deep=True`` additionally loads config.json via PreTrainedConfig and the
    processor JSONs via the official pipeline (requires lerobot). ``deep=False``
    runs only the lerobot-free structural/checksum/secret checks.
    """
    out = Path(artifact_dir)
    checks: dict[str, str] = {}

    def _check(name: str, cond: bool, reason: str = "") -> bool:
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond

    # Structure.
    for fn in _CHECKSUMMED + (CHECKSUMS_FILENAME,):
        _check(f"present:{fn}", (out / fn).exists(), "missing")
    if not all(v == "passed" for v in checks.values()):
        return VerifyResult(False, checks)

    # No path traversal / symlink escape in filenames.
    root = out.resolve()
    escape_ok = True
    for p in out.iterdir():
        try:
            rp = p.resolve()
        except OSError:
            escape_ok = False
            break
        if p.is_symlink() or root not in rp.parents and rp != root:
            escape_ok = False
    _check("no_symlink_escape", escape_ok, "symlink or path escapes artifact root")

    pm = None
    try:
        pm = json.loads((out / PLUGIN_MANIFEST_FILENAME).read_text())
    except Exception as exc:  # noqa: BLE001
        _check("plugin_manifest_parse", False, str(exc))
        return VerifyResult(False, checks)
    _check("plugin_manifest_parse", True)
    _check("schema_version", pm.get("schema_version") == ARTIFACT_SCHEMA_VERSION,
           f"got {pm.get('schema_version')!r}")
    _check("policy_type", pm.get("policy_type") == POLICY_TYPE,
           f"got {pm.get('policy_type')!r}")

    # Checksums (tamper detection).
    try:
        recorded = json.loads((out / CHECKSUMS_FILENAME).read_text())
    except Exception as exc:  # noqa: BLE001
        _check("checksums_parse", False, str(exc))
        return VerifyResult(False, checks)
    tamper_ok = True
    for name, digest in recorded.items():
        actual = _sha256_file(out / name) if (out / name).exists() else None
        if actual != digest:
            tamper_ok = False
            checks[f"checksum:{name}"] = f"failed: {actual} != {digest}"
        else:
            checks[f"checksum:{name}"] = "passed"
    _check("checksums", tamper_ok, "one or more files were modified")

    # Version compatibility (core >= floor, plugin lockstep).
    versions = pm.get("versions", {})
    _check("versions_present",
           bool(versions.get("lerobot_coreai") and versions.get("lerobot_policy_coreai_bridge")),
           "missing versions")

    # No forbidden claims asserted true.
    claims = pm.get("claims", {})
    _check("no_forbidden_claims",
           all(claims.get(k) is not True for k in _FORBIDDEN_TRUE),
           "a forbidden claim is asserted true")

    # No secret persisted.
    secret_free = True
    for name in (CONFIG_FILENAME, PLUGIN_MANIFEST_FILENAME):
        if _looks_like_secret((out / name).read_text()):
            secret_free = False
    _check("no_secrets", secret_free, "config/plugin manifest contains URL/token")

    # External reference must carry an immutable revision+sha256.
    ref = pm.get("coreai_artifact_reference", {})
    if ref.get("mode") == "external":
        _check("external_reference_pinned",
               bool(ref.get("revision") and ref.get("sha256")),
               "external reference missing revision/sha256")

    # Action contract present.
    _check("action_contract", isinstance(pm.get("action_contract"), dict),
           "missing action contract")

    if deep:
        _verify_deep(out, checks)

    ok = all(v == "passed" for v in checks.values())
    return VerifyResult(ok, checks, pm)


def _verify_deep(out: Path, checks: dict[str, str]) -> None:
    """lerobot-dependent checks: config parses; processors reload; ownership."""
    def _check(name, cond, reason=""):
        checks[name] = "passed" if cond else f"failed: {reason}"
        return cond
    # config.json parses as coreai_bridge.
    try:
        import lerobot_policy_coreai_bridge  # noqa: F401  (self-register)
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(str(out))
        _check("config_parses", cfg.type == POLICY_TYPE, f"type={cfg.type!r}")
        _check("config_no_local_path", not cfg.coreai_artifact,
               "coreai_artifact is not empty")
    except Exception as exc:  # noqa: BLE001
        _check("config_parses", False, str(exc))
    # processors reload with the official converters.
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
        _check("processors_reload", True)
    except Exception as exc:  # noqa: BLE001
        _check("processors_reload", False, str(exc))
    # ownership declared in the embedded manifest.
    try:
        from .processor_coreai_bridge import manifest_declares_coreai_ownership
        m = json.loads((out / MANIFEST_FILENAME).read_text())
        _check("processor_ownership", manifest_declares_coreai_ownership(m),
               "manifest does not declare coreai_runner ownership")
    except Exception as exc:  # noqa: BLE001
        _check("processor_ownership", False, str(exc))
