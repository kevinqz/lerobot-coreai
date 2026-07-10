# test_e2e_official_rollout.py — official LeRobot rollout, hardened (v1.3.12).
#
# Drives the REAL lerobot.scripts.lerobot_eval.rollout over a deterministic
# gym.vector.SyncVectorEnv with a COMMON _max_episode_steps and staggered
# terminate_at, HORIZON=3 so the temporal queue drains and refills. Asserts exact
# request counts, cumulative done masks, staggered termination, and the wire
# payload (batched obs, both cameras, no label leakage). Emits a schema-valid
# readiness report. Nothing patched. Not eval/task-success/safety.
#
# Runs where the full LeRobot media stack is present (locally + the dedicated CI
# rollout job). COREAI_REQUIRE_ROLLOUT=1 turns a missing dependency into a FAILURE
# (the mandatory gate) instead of a skip.

import json
import math
import os
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

from lerobot_coreai.runner import capabilities_sha256  # noqa: E402
from lerobot_policy_coreai_bridge import build_plugin_artifact  # noqa: E402
from lerobot_policy_coreai_bridge.rollout_evidence import (  # noqa: E402
    RolloutMeasurements, _sha256_file, build_rollout_readiness_report,
    evaluate_rollout_measurements, write_evidence_bundle,
)

_REQUIRED_OBS = ("observation.state", "observation.images.front",
                 "observation.images.wrist")

_REQUIRE = os.environ.get("COREAI_REQUIRE_ROLLOUT") == "1"
try:
    from lerobot.scripts.lerobot_eval import rollout  # noqa: E402
except ImportError as _exc:  # pragma: no cover
    if _REQUIRE:
        raise
    pytest.skip(f"lerobot_eval unavailable ({_exc}); set COREAI_REQUIRE_ROLLOUT=1 "
                "in the dedicated CI job to make this mandatory",
                allow_module_level=True)

HORIZON, A, C, H, W = 3, 7, 3, 8, 8
MAX_STEPS = 8
_LEAK_KEYS = ("action", "reward", "done", "success", "index", "episode_index",
              "frame_index", "timestamp")


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
    def __init__(self, terminate_at):
        self.terminate_at = terminate_at
        self.t = 0
        self._max_episode_steps = MAX_STEPS      # common max across envs (P1.2)
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(-1, 1, (A,), np.float32),
            "pixels": spaces.Dict({
                "front": spaces.Box(0, 255, (H, W, C), np.uint8),
                "wrist": spaces.Box(0, 255, (H, W, C), np.uint8)})})
        self.action_space = spaces.Box(-1, 1, (A,), np.float32)

    def _obs(self):
        return {"agent_pos": np.zeros(A, np.float32),
                "pixels": {"front": np.zeros((H, W, C), np.uint8),
                           "wrist": np.zeros((H, W, C), np.uint8)}}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, self.t >= self.terminate_at, False, {}

    def task_description(self):
        return "push the T"

    task = "push the T"


class _State:
    def __init__(self, native):
        self.native = native
        self.n = 0
        self.bodies = []


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
            state.bodies.append(body)
            state.n += 1
            opts = body.get("options", {})
            if "batch_size" in opts:
                b = int(opts["batch_size"])
                action = [[[0.0] * A for _ in range(HORIZON)] for _ in range(b)]
            else:
                action = [[0.0] * A for _ in range(HORIZON)]
            return self._j(200, {"action": action})
    return Hd


class _Server:
    def __init__(self, state):
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
        self.port = self._srv.server_address[1]

    def __enter__(self):
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown()
        self._srv.server_close()

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
    src = tmp_path / "coreai"
    src.mkdir()
    (src / "lerobot-coreai.json").write_text(json.dumps(_manifest()))
    art = tmp_path / "plugin"
    build_plugin_artifact(str(src), str(art), runner_url_env="COREAI_RUNNER_URL")
    state = _State(native)
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(str(art))
    cfg.pretrained_path = str(art)
    cfg.batch_mode = batch_mode
    policy = make_policy(cfg, ds_meta=_ds_meta())
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(art))
    venv = gym.vector.SyncVectorEnv([lambda t=t: _Env(t) for t in terminate_ats])
    return policy, pre, post, venv, state, server, art


def _run(policy, pre, post, venv):
    return rollout(venv, policy, _env_pipeline("env_pre"), _env_pipeline("env_post"),
                   pre, post, seeds=list(range(venv.num_envs)))


def _assert_done_mask(done, terminate_ats):
    d = done.int()
    for i, ta in enumerate(terminate_ats):
        fd = ta - 1
        assert not d[i, :fd].any(), f"env {i} done before terminate_at"
        assert d[i, fd:].all(), f"env {i} done not cumulative from {fd}"


def _assert_wire(bodies, native, B):
    assert bodies, "runner received no requests"
    for body in bodies:
        obs = body["observation"]
        assert set(obs.keys()) == {"observation.state", "observation.images.front",
                                   "observation.images.wrist", "task"}
        for leak in _LEAK_KEYS:
            assert leak not in obs, f"label {leak!r} leaked to the runner"
        if native and B > 1:
            assert body["options"]["batch_size"] == B
            assert len(obs["observation.state"]) == B and len(obs["observation.state"][0]) == A
            # images are [B, C, H, W] (fixture feature semantics, P1.9)
            fr = obs["observation.images.front"]
            assert len(fr) == B and len(fr[0]) == C and len(fr[0][0]) == H \
                and len(fr[0][0][0]) == W
            assert isinstance(obs["task"], list) and len(obs["task"]) == B
        else:
            assert "batch_size" not in body.get("options", {}) or B == 1
            assert len(obs["observation.state"]) == A          # single sample
            fr = obs["observation.images.front"]
            assert len(fr) == C and len(fr[0]) == H and len(fr[0][0]) == W
            assert isinstance(obs["task"], str)


def _evidence(tmp_path, out, state, art, policy, B, mode, terminate_ats):
    """Build measurements -> evaluate -> real-hash report -> persisted bundle."""
    seq = out["action"].shape[1]
    inv = json.loads((art / "plugin_artifact_inventory.json").read_text())
    pm = json.loads((art / "plugin_artifact_manifest.json").read_text())
    caps_sha = (capabilities_sha256(policy._capabilities)
                if policy._capabilities is not None else "sha256:" + "e" * 64)
    m = RolloutMeasurements(
        batch_size=B, mode=mode, sequence_length=seq, horizon=HORIZON,
        terminate_at=tuple(terminate_ats), request_bodies=tuple(state.bodies),
        done_mask=tuple(tuple(int(x) for x in row) for row in out["done"].int().tolist()),
        required_obs_keys=_REQUIRED_OBS)
    ev = evaluate_rollout_measurements(m)
    report = build_rollout_readiness_report(
        ev, m, environment={"lerobot": "0.6.x"},
        artifact_root_sha256=inv["artifact_root_sha256"],
        batch_contract_sha256=pm["batch_contract_sha256"],
        runner_capabilities_sha256=caps_sha,
        preprocessor_sha256=_sha256_file(art / "policy_preprocessor.json"),
        postprocessor_sha256=_sha256_file(art / "policy_postprocessor.json"))
    bundle = tmp_path / "evidence" / f"{mode}-b{B}"
    write_evidence_bundle(str(bundle), report, m)
    assert (bundle / "official_rollout_readiness_report.json").exists()
    assert (bundle / "checksums.json").exists()
    return ev, report


@pytest.mark.parametrize("B,terminate_ats", [(1, [8]), (2, [3, 8]), (4, [2, 4, 6, 8])])
def test_official_rollout_native(tmp_path, monkeypatch, B, terminate_ats):
    mode = "auto" if B == 1 else "native_batch"
    policy, pre, post, venv, state, server, art = _setup(
        tmp_path, monkeypatch, native=True, batch_mode=mode, terminate_ats=terminate_ats)
    try:
        out = _run(policy, pre, post, venv)
        seq = out["action"].shape[1]
        assert out["action"].shape[0] == B and out["action"].shape[-1] == A
        assert seq == MAX_STEPS
        _assert_done_mask(out["done"], terminate_ats)
        assert state.n == math.ceil(seq / HORIZON)          # native: one per refill
        _assert_wire(state.bodies, native=True, B=B)
        eff_mode = "single_only" if B == 1 else "native_batch"
        ev, report = _evidence(tmp_path, out, state, art, policy, B, eff_mode,
                               terminate_ats)
        assert ev.passed, (ev.failed_stage, ev.errors)
        assert report["claims"]["official_rollout_pipeline_smoke_passed"] is True
        assert report["claims"]["official_eval_certified"] is False
    finally:
        server.__exit__()
        venv.close()


@pytest.mark.parametrize("B,terminate_ats", [(2, [3, 8]), (4, [2, 4, 6, 8])])
def test_official_rollout_split(tmp_path, monkeypatch, B, terminate_ats):
    policy, pre, post, venv, state, server, art = _setup(
        tmp_path, monkeypatch, native=False, batch_mode="split_and_stack",
        terminate_ats=terminate_ats)
    try:
        out = _run(policy, pre, post, venv)
        seq = out["action"].shape[1]
        assert out["action"].shape[0] == B
        _assert_done_mask(out["done"], terminate_ats)
        assert state.n == B * math.ceil(seq / HORIZON)      # split: B per refill
        _assert_wire(state.bodies, native=False, B=B)
        ev, report = _evidence(tmp_path, out, state, art, policy, B,
                               "split_and_stack", terminate_ats)
        assert ev.passed, (ev.failed_stage, ev.errors)
    finally:
        server.__exit__()
        venv.close()
