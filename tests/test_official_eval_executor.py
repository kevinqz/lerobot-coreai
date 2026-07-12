# test_official_eval_executor.py — the executor that OWNS official-eval receipt creation
# (v1.3.27 WS1 foundation). The resolution + real-subprocess tests run wherever lerobot
# is installed (the rollout CI jobs) and skip in the torch-free base jobs; the assembly
# tests are pure and run everywhere.

import importlib.util

import pytest

from lerobot_coreai.authority import (
    AuthorityError, verify_official_eval_execution_receipt,
)
from lerobot_coreai.official_eval_executor import (
    OfficialEvalExecutorError, build_execution_receipt, output_tree_sha256,
    resolve_official_eval_entrypoint, run_official_eval,
)

_HAS_LEROBOT = importlib.util.find_spec("lerobot") is not None
_needs_lerobot = pytest.mark.skipif(not _HAS_LEROBOT, reason="lerobot not installed")
_H = "sha256:" + "a" * 64
_CASES = ["single-b1", "native-b2", "native-b4", "split-b2", "split-b4"]


def _resolved():
    return {"resolution_method": "python_-m",
            "executable_realpath": "/opt/venv/lib/lerobot/scripts/lerobot_eval.py",
            "lerobot_distribution_sha256": _H}


def _run(exit_code=0, echoed=True):
    return {"argv": ["/opt/venv/bin/python", "-m", "lerobot.scripts.lerobot_eval",
                     "--env.type=coreai_cert_env", "--policy.type=coreai_bridge"],
            "exit_code": exit_code, "challenge_echoed": echoed}


def _receipt(**over):
    kw = dict(resolved=_resolved(), run=_run(), cases=list(_CASES),
              env_instantiated=True, output_tree=_H, resolved_config_sha256=_H,
              outputs_schema_valid=True, output_manifest_sha256=_H,
              evidence_replay_passed=True, replay_root_sha256=_H,
              verified_cases_root_sha256=_H)
    kw.update(over)
    return build_execution_receipt(**kw)


# --- pure assembly (runs everywhere) ---

def test_output_tree_sha256_is_content_addressed(tmp_path):
    (tmp_path / "a.json").write_text('{"x":1}')
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("hello")
    first = output_tree_sha256(str(tmp_path))
    assert first.startswith("sha256:")
    (tmp_path / "sub" / "b.txt").write_text("HELLO")     # change bytes
    assert output_tree_sha256(str(tmp_path)) != first    # digest tracks content


def test_receipt_from_real_run_is_certificate_grade():
    # a receipt assembled from a clean, full-matrix, env-instantiated run mints.
    receipt = _receipt()
    assert receipt["real_subprocess"] is True and receipt["fake_executor"] is False
    verified = verify_official_eval_execution_receipt(receipt)
    assert verified.payload["_namespace"] == "test_only"   # declarative-executor grade


def test_partial_run_does_not_certify():
    # a --help / partial run (incomplete matrix) cannot mint a certificate-grade receipt.
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(_receipt(cases=["single-b1"]))


def test_env_not_echoed_blocks_instantiation_flag():
    # if the challenge nonce did not round-trip, coreai_env_instantiated is false → the
    # receipt is refused (the env was not proven to be instantiated).
    r = _receipt(run=_run(echoed=False))
    assert r["coreai_env_instantiated"] is False
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(r)


def test_nonzero_exit_run_refused():
    with pytest.raises(AuthorityError):
        verify_official_eval_execution_receipt(_receipt(run=_run(exit_code=1)))


# --- real resolution + subprocess (rollout jobs only) ---

@_needs_lerobot
def test_resolve_binds_to_installed_lerobot_distribution():
    resolved = resolve_official_eval_entrypoint()
    assert resolved["distribution"] == "lerobot"
    assert resolved["resolution_method"] == "python_-m"
    assert resolved["executable_realpath"].endswith("lerobot_eval.py")
    assert resolved["lerobot_distribution_sha256"].startswith("sha256:")


@_needs_lerobot
def test_run_executes_the_real_entrypoint():
    # genuinely execute the installed official console script (not a declared boolean).
    run = run_official_eval(["--help"], challenge_nonce="nonce-abc12345", timeout=300)
    assert run["exit_code"] == 0
    assert "lerobot.scripts.lerobot_eval" in " ".join(run["argv"])
    assert ("policy" in run["stdout"].lower() or "env" in run["stdout"].lower())


def test_resolve_raises_without_lerobot():
    # in the torch-free base job (no lerobot) resolution must fail closed, not fabricate.
    if _HAS_LEROBOT:
        pytest.skip("lerobot installed")
    with pytest.raises(OfficialEvalExecutorError):
        resolve_official_eval_entrypoint()
