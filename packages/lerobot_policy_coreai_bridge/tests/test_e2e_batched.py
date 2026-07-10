# test_e2e_batched.py — stateless batched runtime B=2/B=4, no mocks (v1.3.9).
#
# Native (one request -> Tensor[B,H,A]) and split-and-stack (B requests -> stacked)
# over the full official chain (make_policy + make_pre_post_processors) against a
# real HTTP runner that announces a stateless, batch-capable protocol. Nothing is
# patched. Proves shape/state/atomic/temporal batching; not eval or safety.

import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")
import torch  # noqa: E402

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from lerobot_coreai.errors import CoreAIPolicyError  # noqa: E402
from lerobot_policy_coreai_bridge import build_plugin_artifact  # noqa: E402
from lerobot_policy_coreai_bridge.modeling_coreai_bridge import PluginBindingError  # noqa: E402

HORIZON, ACTION_DIM = 3, 7


def _manifest():
    return {
        "schema_version": "lerobot-coreai.v0", "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0", "commit": None},
        "policy": {"repo_id": "kevinqz/E2E", "source_repo_id": "lerobot/e2e",
                   "type": "evo1", "class": None, "config_class": None},
        "robot": {"type": "so100", "action_representation": "joint_position_delta",
                  "fps": 30},
        "features": {
            "observation": {
                "observation.state": {"dtype": "float32", "shape": [ACTION_DIM],
                                      "required": True},
                "task": {"dtype": "string", "required": False}},
            "action": {"action": {"dtype": "float32", "shape": [HORIZON, ACTION_DIM]}}},
        "normalization": {"format": "lerobot", "path": "norm_stats.json", "sha256": None},
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
            "processor": {
                "observation_input": {"owner": "coreai_runner",
                                      "expects": "raw_lerobot_observation"},
                "action_output": {"owner": "coreai_runner",
                                  "returns": "postprocessed_environment_action"}},
            "batch": {"schema_version": "coreai-batch-contract.v2",
                      "policy_supports_batch": True,
                      "supported_client_modes": ["native_batch", "split_and_stack"],
                      "max_batch_size": 4, "fallback": "split_and_stack",
                      "queue_layout": "time_major_batched",
                      "requires_atomic_commit": True}},
    }


class _State:
    def __init__(self, *, native: bool, bad_sample: int | None = None):
        self.native = native
        self.bad_sample = bad_sample     # split: return wrong-dim action at this index
        self.n = 0
        self.bodies = []


def _handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _j(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/v1/health":
                return self._j(200, {"status": "ok"})
            if self.path == "/v1/capabilities":
                caps = {
                    "runtime": "coreai-runner", "supports": {"action": True},
                    "protocol_version": "coreai-runner.v2",
                    "observation_encodings": ["nested_json_v1"],
                    "inference_state": {"scope": "stateless",
                                        "supports_session_ids": False,
                                        "reset_scope": "none"}}
                caps["action_batching"] = (
                    {"supported": True, "max_batch_size": 4, "semantics": "native",
                     "state_isolation": "stateless"} if state.native
                    else {"supported": False, "max_batch_size": 4,
                          "semantics": "split_and_stack", "state_isolation": "stateless"})
                return self._j(200, caps)
            return self._j(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            idx = state.n
            state.bodies.append(body)
            state.n += 1
            opts = body.get("options", {})
            if "batch_size" in opts:                  # native batched request
                b = int(opts["batch_size"])
                action = [[[float(i)] * ACTION_DIM for _ in range(HORIZON)]
                          for i in range(b)]          # [B,H,A]
            else:                                     # single request (B=1 or split)
                dim = 99 if state.bad_sample == idx else ACTION_DIM
                action = [[0.0] * dim for _ in range(HORIZON)]    # [H,A]
            return self._j(200, {"action": action})
    return H


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
    return types.SimpleNamespace(
        features={"observation.state": {"dtype": "float32", "shape": [ACTION_DIM]},
                  "action": {"dtype": "float32", "shape": [ACTION_DIM]}},
        stats={})


def _policy(tmp_path, monkeypatch, state, *, batch_mode):
    src = tmp_path / "coreai"
    src.mkdir()
    (src / "lerobot-coreai.json").write_text(json.dumps(_manifest()))
    art = tmp_path / "plugin"
    build_plugin_artifact(str(src), str(art), runner_url_env="COREAI_RUNNER_URL")
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(str(art))
    cfg.pretrained_path = str(art)
    cfg.batch_mode = batch_mode
    policy = make_policy(cfg, ds_meta=_ds_meta())
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(art))
    return policy, pre, post, server


def _batch(B):
    return {"observation.state": torch.zeros(B, ACTION_DIM),
            "task": [f"task {i}" for i in range(B)]}


# --- native ---

@pytest.mark.parametrize("B", [2, 4])
def test_native_batch_one_request(tmp_path, monkeypatch, B):
    state = _State(native=True)
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="native_batch")
    try:
        chunk = policy.predict_action_chunk(_batch(B))
        assert tuple(chunk.shape) == (B, HORIZON, ACTION_DIM)
        assert state.n == 1                                  # ONE request
        action = policy.select_action(_batch(B))
        assert tuple(action.shape) == (B, ACTION_DIM)
    finally:
        server.__exit__()


def test_native_wire_has_batched_obs_and_no_leak(tmp_path, monkeypatch):
    state = _State(native=True)
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="native_batch")
    try:
        policy.predict_action_chunk(_batch(2))
        obs = state.bodies[0]["observation"]
        assert set(obs.keys()) == {"observation.state", "task"}
        assert len(obs["observation.state"]) == 2 and len(obs["observation.state"][0]) == ACTION_DIM
        assert obs["task"] == ["task 0", "task 1"]
        assert state.bodies[0]["options"]["batch_size"] == 2
    finally:
        server.__exit__()


# --- split ---

@pytest.mark.parametrize("B", [2, 4])
def test_split_batch_b_requests(tmp_path, monkeypatch, B):
    state = _State(native=False)
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="split_and_stack")
    try:
        chunk = policy.predict_action_chunk(_batch(B))
        assert tuple(chunk.shape) == (B, HORIZON, ACTION_DIM)
        assert state.n == B                                  # B requests
        # each split request carried a SINGLE observation.
        assert len(state.bodies[0]["observation"]["observation.state"]) == ACTION_DIM
    finally:
        server.__exit__()


def test_split_atomic_rollback_on_sample_failure(tmp_path, monkeypatch):
    state = _State(native=False, bad_sample=2)            # sample 2 -> wrong dim
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="split_and_stack")
    try:
        with pytest.raises(PluginBindingError) as ei:
            policy.predict_action_chunk(_batch(4))
        assert "sample index 2" in str(ei.value)
        assert len(policy._queue) == 0                    # queue untouched
    finally:
        server.__exit__()


# --- temporal queue + guards ---

def test_temporal_queue_drains_without_extra_requests(tmp_path, monkeypatch):
    state = _State(native=True)
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="native_batch")
    try:
        for _ in range(HORIZON):                          # drain a full chunk
            a = policy.select_action(_batch(2))
            assert tuple(a.shape) == (2, ACTION_DIM)
        assert state.n == 1                               # one predict for H steps
    finally:
        server.__exit__()


def test_batch_size_change_while_queue_nonempty_fails(tmp_path, monkeypatch):
    state = _State(native=True)
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="native_batch")
    try:
        policy.select_action(_batch(2))                   # fills queue with B=2
        with pytest.raises(PluginBindingError):
            policy.select_action(_batch(4))               # B changed mid-queue
    finally:
        server.__exit__()


def test_b1_still_works_backward_compatible(tmp_path, monkeypatch):
    state = _State(native=True)
    policy, pre, post, server = _policy(tmp_path, monkeypatch, state,
                                        batch_mode="auto")
    try:
        a = policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM),
                                  "task": ["one"]})
        assert tuple(a.shape) == (1, ACTION_DIM)
    finally:
        server.__exit__()
