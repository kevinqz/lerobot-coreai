# test_e2e_official_rollout.py — verifiable official-rollout evidence (v1.3.14).
#
# Drives the REAL lerobot_eval.rollout, captures request AND response/action data,
# binds exact environment identity + deep-verified artifact hashes, writes a per-case
# evidence bundle + an aggregate matrix, and re-proves the whole thing with the
# OFFLINE verifier (tamper/reorder/missing-case detection). Time/slot-dependent
# fixture so ordered request hashes are genuinely distinct. Nothing patched.

import json
import math
import os
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")
gym = pytest.importorskip("gymnasium")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from gymnasium import spaces  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.processor import (  # noqa: E402
    PolicyProcessorPipeline, batch_to_transition, transition_to_batch,
)
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from lerobot_coreai.rollout_verify import verify_official_rollout_evidence  # noqa: E402
from lerobot_coreai.runner import capabilities_sha256  # noqa: E402
from lerobot_policy_coreai_bridge import (  # noqa: E402
    build_plugin_artifact, verify_plugin_artifact,
)
from lerobot_policy_coreai_bridge.rollout_evidence import (  # noqa: E402
    EvidenceBindingError, RolloutMeasurements, _sha256_file,
    build_rollout_readiness_report, capture_environment_identity,
    evaluate_rollout_measurements, write_evidence_bundle, write_matrix_manifest,
)

_REQUIRE = os.environ.get("COREAI_REQUIRE_ROLLOUT") == "1"
try:
    from lerobot.scripts.lerobot_eval import rollout  # noqa: E402
except ImportError as _exc:  # pragma: no cover
    if _REQUIRE:
        raise
    pytest.skip(f"lerobot_eval unavailable ({_exc})", allow_module_level=True)

HORIZON, A, C, H, W = 3, 7, 3, 8, 8
MAX_STEPS = 8
_FIXTURE = {"observation.state": [A], "observation.images.front": [C, H, W],
            "observation.images.wrist": [C, H, W]}


def _manifest():
    return {
        "schema_version": "lerobot-coreai.v0", "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0", "commit": None},
        "policy": {"repo_id": "k/E", "source_repo_id": "l/e", "type": "evo1",
                   "class": None, "config_class": None},
        "robot": {"type": "so100", "action_representation": "joint_position_delta",
                  "fps": 30},
        "features": {"observation": {
            "observation.state": {"dtype": "float32", "shape": [A], "required": True},
            "observation.images.front": {"dtype": "image", "shape": [C, H, W],
                                         "required": True},
            "observation.images.wrist": {"dtype": "image", "shape": [C, H, W],
                                         "required": True},
            "task": {"dtype": "string", "required": False}},
            "action": {"action": {"dtype": "float32", "shape": [HORIZON, A]}}},
        "normalization": {"format": "lerobot", "path": "n.json", "sha256": None},
        "coreai": {"artifact_format": "aimodel", "runner": "coreai-runner",
                   "graphs": [{"name": "g", "role": "denoise_step"}],
                   "host_loop_required": False},
        "evaluation": {"metric": "action_parity", "status": "passed", "n_obs": 8,
                       "min_chunk_cosine": 0.9999, "max_action_mae": None,
                       "max_relative_action_mae": None, "proves_numeric_fidelity": True,
                       "proves_task_success": False, "proves_robot_safety": False},
        "safety": {"default_mode": "dry_run",
                   "real_actuation_requires_confirmation": True},
        "contracts": {
            "processor": {"observation_input": {"owner": "coreai_runner",
                                                "expects": "raw_lerobot_observation"},
                          "action_output": {"owner": "coreai_runner",
                                            "returns": "postprocessed_environment_action"}},
            "batch": {"schema_version": "coreai-batch-contract.v3",
                      "native_batch": {"supported": True, "max_batch_size": 4,
                                       "required_slot_isolation": "independent"},
                      "client_split": {"supported": True, "max_batch_size": 4,
                                       "allowed_state_scopes": ["stateless"]},
                      "fallback": "split_and_stack",
                      "queue": {"layout": "time_major_batched",
                                "commit_semantics": "atomic_queue_commit"},
                      "observation_stage": "lerobot_policy_preprocessor_output.v1"}},
    }


class _Env(gym.Env):
    def __init__(self, terminate_at, slot):
        self.terminate_at, self.slot, self.t = terminate_at, slot, 0
        self._max_episode_steps = MAX_STEPS
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(-1e9, 1e9, (A,), np.float32),
            "pixels": spaces.Dict({
                "front": spaces.Box(0, 255, (H, W, C), np.uint8),
                "wrist": spaces.Box(0, 255, (H, W, C), np.uint8)})})
        self.action_space = spaces.Box(-1, 1, (A,), np.float32)

    def _obs(self):
        # time- and slot-dependent so consecutive/parallel requests differ (P1.10)
        st = np.zeros(A, np.float32); st[0] = float(self.t); st[1] = float(self.slot)
        fr = np.zeros((H, W, C), np.uint8); fr[0, 0, 0] = self.t % 256
        wr = np.zeros((H, W, C), np.uint8); wr[0, 0, 0] = self.slot % 256
        return {"agent_pos": st, "pixels": {"front": fr, "wrist": wr}}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, self.t >= self.terminate_at, False, {}

    def task_description(self):
        return "push the T"

    task = "push the T"


class _State:
    def __init__(self, native):
        self.native, self.n = native, 0
        self.bodies, self.responses = [], []


def _handler(state):
    class Hd(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _j(self, c, o):
            b = json.dumps(o).encode()
            self.send_response(c)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/v1/health":
                return self._j(200, {"status": "ok"})
            if self.path == "/v1/capabilities":
                ab = ({"supported": True, "max_batch_size": 4, "semantics": "native",
                       "slot_isolation": "independent"} if state.native
                      else {"supported": False, "semantics": "split_and_stack",
                            "slot_isolation": "independent"})
                return self._j(200, {"runtime": "coreai-runner",
                                     "supports": {"action": True},
                                     "protocol_version": "coreai-runner.v2",
                                     "observation_encodings": ["nested_json_v1"],
                                     "action_batching": ab,
                                     "inference_state": {"scope": "stateless",
                                                         "supports_session_ids": False,
                                                         "reset_scope": "none"}})
            return self._j(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            idx = state.n            # deterministic, index-dependent (WS E, P1.5)
            state.bodies.append(body)
            state.n += 1
            opts = body.get("options", {})
            if "batch_size" in opts:
                b = int(opts["batch_size"])
                action = [[[float(idx * 10000 + s * 1000 + h * 100 + a)
                            for a in range(A)] for h in range(HORIZON)]
                          for s in range(b)]
            else:
                action = [[float(idx * 10000 + h * 100 + a) for a in range(A)]
                          for h in range(HORIZON)]
            resp = {"action": action}
            state.responses.append(resp)
            return self._j(200, resp)
    return Hd


class _Server:
    def __init__(self, state):
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
        self.port = self._srv.server_address[1]

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown(); self._srv.server_close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def _ds_meta():
    return types.SimpleNamespace(features={
        "observation.state": {"dtype": "float32", "shape": [A]},
        "observation.images.front": {"dtype": "image", "shape": [C, H, W],
                                     "names": ["channel", "height", "width"]},
        "observation.images.wrist": {"dtype": "image", "shape": [C, H, W],
                                     "names": ["channel", "height", "width"]},
        "action": {"dtype": "float32", "shape": [A]}}, stats={})


def _env_pipeline(name):
    return PolicyProcessorPipeline(steps=[], name=name,
                                   to_transition=batch_to_transition,
                                   to_output=transition_to_batch)


def _setup(tmp_path, monkeypatch, native, batch_mode, terminate_ats):
    src = tmp_path / "coreai"; src.mkdir(exist_ok=True)
    (src / "lerobot-coreai.json").write_text(json.dumps(_manifest()))
    art = tmp_path / "plugin"
    build_plugin_artifact(str(src), str(art), runner_url_env="COREAI_RUNNER_URL")
    state = _State(native)
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(str(art))
    cfg.pretrained_path = str(art); cfg.batch_mode = batch_mode
    policy = make_policy(cfg, ds_meta=_ds_meta())
    policy.begin_evidence_session("rollout")   # v1.3.18: causal evidence session
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(art))
    venv = gym.vector.SyncVectorEnv(
        [lambda t=t, s=s: _Env(t, s) for s, t in enumerate(terminate_ats)])
    return policy, pre, post, venv, state, server, art


def _evidence(evidence_dir, out, state, art, policy, B, mode, terminate_ats):
    policy.end_evidence_session()                         # seal the causal event stream
    if policy._capabilities is None:                      # P1.2: no fallback
        raise EvidenceBindingError("evidence requires negotiated Runner capabilities")
    verified = verify_plugin_artifact(str(art), deep=True)   # P1.3: recompute
    assert verified.ok, verified.checks
    inv = json.loads((art / "plugin_artifact_inventory.json").read_text())
    pm = json.loads((art / "plugin_artifact_manifest.json").read_text())
    target = os.environ.get("COREAI_ROLLOUT_TARGET", "local")
    m = RolloutMeasurements(
        batch_size=B, mode=mode, sequence_length=out["action"].shape[1], horizon=HORIZON,
        action_dim=A, terminate_at=tuple(terminate_ats),
        request_bodies=tuple(state.bodies), response_bodies=tuple(state.responses),
        done_mask=tuple(tuple(int(x) for x in r) for r in out["done"].int().tolist()),
        final_action=out["action"].to("cpu").tolist(), required_obs_keys=tuple(_FIXTURE),
        fixture_contract=_FIXTURE, queue_events=tuple(policy.queue_events),
        negotiation=policy.negotiation_record,
        runner_capabilities=policy.runner_capabilities_raw,
        runner_capabilities_normalized=policy.runner_capabilities_normalized)
    ev = evaluate_rollout_measurements(m)
    report = build_rollout_readiness_report(
        ev, m, environment=capture_environment_identity(target),
        artifact_root_sha256=inv["artifact_root_sha256"],
        batch_contract_sha256=pm["batch_contract_sha256"],
        runner_capabilities_sha256=capabilities_sha256(policy._capabilities),
        preprocessor_sha256=_sha256_file(art / "policy_preprocessor.json"),
        postprocessor_sha256=_sha256_file(art / "policy_postprocessor.json"),
        artifact_integrity_verified=verified.ok)
    case = f"{mode}-b{B}"
    root = write_evidence_bundle(str(Path(evidence_dir) / case), report, m)
    return ev, report, root


# --- granular ---

@pytest.mark.parametrize("B,terminate_ats", [(1, [8]), (2, [3, 8]), (4, [2, 4, 6, 8])])
def test_native(tmp_path, monkeypatch, B, terminate_ats):
    mode = "auto" if B == 1 else "native_batch"
    policy, pre, post, venv, state, server, art = _setup(
        tmp_path, monkeypatch, True, mode, terminate_ats)
    try:
        out = rollout(venv, policy, _env_pipeline("ep"), _env_pipeline("eo"), pre, post,
                      seeds=list(range(B)))
        assert state.n == math.ceil(out["action"].shape[1] / HORIZON)
        ev, report, _ = _evidence(tmp_path / "ev", out, state, art, policy, B,
                                  "single_only" if B == 1 else "native_batch", terminate_ats)
        assert ev.passed, (ev.failed_stage, ev.errors)
        assert report["observation"]["distinct_request_hashes"] is True
        assert report["contracts"]["artifact_integrity_verified"] is True
    finally:
        server.__exit__(); venv.close()


@pytest.mark.parametrize("B,terminate_ats", [(2, [3, 8]), (4, [2, 4, 6, 8])])
def test_split(tmp_path, monkeypatch, B, terminate_ats):
    policy, pre, post, venv, state, server, art = _setup(
        tmp_path, monkeypatch, False, "split_and_stack", terminate_ats)
    try:
        out = rollout(venv, policy, _env_pipeline("ep"), _env_pipeline("eo"), pre, post,
                      seeds=list(range(B)))
        assert state.n == B * math.ceil(out["action"].shape[1] / HORIZON)
        ev, _, _ = _evidence(tmp_path / "ev", out, state, art, policy, B,
                             "split_and_stack", terminate_ats)
        assert ev.passed, (ev.failed_stage, ev.errors)
    finally:
        server.__exit__(); venv.close()


# --- full matrix + OFFLINE independent verification ---

def test_matrix_and_offline_verify(tmp_path, monkeypatch):
    # CI points COREAI_ROLLOUT_EVIDENCE_DIR here so the matrix bundle is uploaded.
    ev_dir = Path(os.environ.get("COREAI_ROLLOUT_EVIDENCE_DIR") or (tmp_path / "matrix"))
    cases_meta: dict[str, dict] = {}
    configs = [(True, "auto", [8], 1, "single_only-b1"),
               (True, "native_batch", [3, 8], 2, "native_batch-b2"),
               (True, "native_batch", [2, 4, 6, 8], 4, "native_batch-b4"),
               (False, "split_and_stack", [3, 8], 2, "split_and_stack-b2"),
               (False, "split_and_stack", [2, 4, 6, 8], 4, "split_and_stack-b4")]
    for native, mode, tats, B, case in configs:
        wd = tmp_path / f"work-{case}"; wd.mkdir()
        policy, pre, post, venv, state, server, art = _setup(
            wd, monkeypatch, native, mode, tats)
        try:
            out = rollout(venv, policy, _env_pipeline("ep"), _env_pipeline("eo"),
                          pre, post, seeds=list(range(B)))
            report_mode = "single_only" if B == 1 else mode
            ev, _, root = _evidence(ev_dir, out, state, art, policy, B, report_mode, tats)
            cases_meta[case] = {"passed": ev.passed, "bundle_root_sha256": root}
        finally:
            server.__exit__(); venv.close()
    write_matrix_manifest(str(ev_dir), os.environ.get("COREAI_ROLLOUT_TARGET", "local"), cases_meta)

    res = verify_official_rollout_evidence(str(ev_dir), require_complete_matrix=True)
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}

    # tamper on a COPY (leave the real bundle clean for CI upload + re-verify).
    import shutil
    tampered = tmp_path / "tampered"
    shutil.copytree(ev_dir, tampered)
    rp = tampered / "native_batch-b2" / "official_rollout_readiness_report.json"
    d = json.loads(rp.read_text()); d["execution"]["request_count"] = 999
    rp.write_text(json.dumps(d))
    assert not verify_official_rollout_evidence(str(tampered)).ok

    # v1.3.20: tampering the persisted runner capabilities breaks the recomputed
    # capabilities hash bound in the NegotiationRecord.
    tampered2 = tmp_path / "tampered-caps"
    shutil.copytree(ev_dir, tampered2)
    cp = tampered2 / "single_only-b1" / "runner_capabilities_raw.json"
    c = json.loads(cp.read_text()); c["protocol_version"] = "coreai-runner.v9"
    cp.write_text(json.dumps(c))
    assert not verify_official_rollout_evidence(str(tampered2)).ok


def test_missing_case_fails_verify(tmp_path, monkeypatch):
    ev_dir = tmp_path / "m"
    wd = tmp_path / "w"; wd.mkdir()
    policy, pre, post, venv, state, server, art = _setup(
        wd, monkeypatch, True, "native_batch", [3, 8])
    try:
        out = rollout(venv, policy, _env_pipeline("ep"), _env_pipeline("eo"), pre, post,
                      seeds=[0, 1])
        _, _, root = _evidence(ev_dir, out, state, art, policy, 2, "native_batch", [3, 8])
    finally:
        server.__exit__(); venv.close()
    write_matrix_manifest(str(ev_dir), os.environ.get("COREAI_ROLLOUT_TARGET", "local"),
                          {"native_batch-b2": {"passed": True, "bundle_root_sha256": root}})
    # only one case present -> incomplete matrix fails.
    assert not verify_official_rollout_evidence(str(ev_dir),
                                                require_complete_matrix=True).ok
