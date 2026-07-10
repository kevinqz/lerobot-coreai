# test_e2e_official_rollout.py — official LeRobot rollout readiness, no mocks (v1.3.11).
#
# Drives the REAL lerobot.scripts.lerobot_eval.rollout over a deterministic
# gym.vector.SyncVectorEnv (state + front/wrist cameras + task_description), with
# staggered episode termination, through the official chain:
#   preprocess_observation -> env_preprocessor -> policy_preprocessor ->
#   CoreAIBridgePolicy.select_action -> policy_postprocessor -> env_postprocessor.
# Native and split modes. Nothing is patched. Proves the official rollout PIPELINE
# completes and yields Tensor[B, seq, A] — NOT task success, safety, or lerobot-eval
# certification.

import json
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

# lerobot_eval imports `datasets` at module load; skip cleanly if it's absent
# (the CI lerobot jobs install it explicitly so these tests actually run).
rollout = pytest.importorskip("lerobot.scripts.lerobot_eval").rollout

from lerobot_policy_coreai_bridge import build_plugin_artifact  # noqa: E402

HORIZON, A, C, H, W = 3, 7, 3, 8, 8


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
    def __init__(self, ttl):
        self.ttl = ttl
        self.t = 0
        self._max_episode_steps = ttl
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
        return self._obs(), 0.0, self.t >= self.ttl, False, {}

    def task_description(self):
        return "push the T"

    task = "push the T"


class _State:
    def __init__(self, native):
        self.native = native
        self.n = 0


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


def _setup(tmp_path, monkeypatch, native, batch_mode, ttls):
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
    venv = gym.vector.SyncVectorEnv([lambda t=t: _Env(t) for t in ttls])
    return policy, pre, post, venv, state, server


def _run(policy, pre, post, venv):
    out = rollout(venv, policy, _env_pipeline("env_pre"), _env_pipeline("env_post"),
                  pre, post, seeds=list(range(venv.num_envs)))
    return out


@pytest.mark.parametrize("B", [1, 2, 4])
def test_official_rollout_native(tmp_path, monkeypatch, B):
    ttls = tuple(range(2, 2 + B))          # staggered termination
    policy, pre, post, venv, state, server = _setup(
        tmp_path, monkeypatch, native=True, batch_mode="auto", ttls=ttls)
    try:
        out = _run(policy, pre, post, venv)
        assert set(out) >= {"action", "reward", "success", "done"}
        act = out["action"]
        assert act.ndim == 3 and act.shape[0] == B and act.shape[-1] == A
        assert state.n >= 1                 # the runner was actually driven
    finally:
        server.__exit__()
        venv.close()


@pytest.mark.parametrize("B", [2, 4])
def test_official_rollout_split(tmp_path, monkeypatch, B):
    ttls = tuple(range(2, 2 + B))
    policy, pre, post, venv, state, server = _setup(
        tmp_path, monkeypatch, native=False, batch_mode="split_and_stack", ttls=ttls)
    try:
        out = _run(policy, pre, post, venv)
        assert out["action"].shape[0] == B and out["action"].shape[-1] == A
        assert state.n >= B                 # split issues >= B requests
    finally:
        server.__exit__()
        venv.close()
