# test_e2e_multimodal.py — multimodal B=1/2/4 through EXECUTED processors (v1.3.10).
#
# state + two cameras (front/wrist) + task, run through the real official chain
# pre(batch) -> policy.select_action -> post(action) against a real batch-capable
# HTTP runner, native and split. Proves the batched multimodal data plane composes
# with processors actually executed (not just loaded). Not eval, not safety.

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

from lerobot_policy_coreai_bridge import build_plugin_artifact  # noqa: E402

HORIZON, ACTION_DIM = 3, 7
C, H, W = 3, 8, 8


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
                "observation.images.front": {"dtype": "image", "shape": [C, H, W],
                                             "required": True},
                "observation.images.wrist": {"dtype": "image", "shape": [C, H, W],
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


class _State:
    def __init__(self, native):
        self.native = native
        self.n = 0
        self.bodies = []


def _handler(state):
    class Hd(BaseHTTPRequestHandler):
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
                ab = ({"supported": True, "max_batch_size": 4, "semantics": "native",
                       "slot_isolation": "independent"} if state.native
                      else {"supported": False, "semantics": "split_and_stack",
                            "slot_isolation": "independent"})
                return self._j(200, {
                    "runtime": "coreai-runner", "supports": {"action": True},
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
                action = [[[0.0] * ACTION_DIM for _ in range(HORIZON)] for _ in range(b)]
            else:
                action = [[0.0] * ACTION_DIM for _ in range(HORIZON)]
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
    return types.SimpleNamespace(
        features={
            "observation.state": {"dtype": "float32", "shape": [ACTION_DIM]},
            "observation.images.front": {"dtype": "image", "shape": [C, H, W],
                                         "names": ["channel", "height", "width"]},
            "observation.images.wrist": {"dtype": "image", "shape": [C, H, W],
                                         "names": ["channel", "height", "width"]},
            "action": {"dtype": "float32", "shape": [ACTION_DIM]}},
        stats={})


def _compose(tmp_path, monkeypatch, state, batch_mode):
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
            "observation.images.front": torch.zeros(B, C, H, W),
            "observation.images.wrist": torch.zeros(B, C, H, W),
            "task": [f"task {i}" for i in range(B)]}


@pytest.mark.parametrize("B", [1, 2, 4])
def test_multimodal_native_executes_processors(tmp_path, monkeypatch, B):
    state = _State(native=True)
    policy, pre, post, server = _compose(tmp_path, monkeypatch, state, "auto")
    try:
        processed = pre(_batch(B))              # processors ACTUALLY executed
        action = policy.select_action(processed)
        final = post(action)
        assert isinstance(final, torch.Tensor) and tuple(final.shape) == (B, ACTION_DIM)
        if B > 1:
            assert state.n == 1                 # native = one request
        # both cameras reached the runner; no ground-truth leakage.
        obs = state.bodies[0]["observation"]
        assert "observation.images.front" in obs and "observation.images.wrist" in obs
        assert "action" not in obs
    finally:
        server.__exit__()


@pytest.mark.parametrize("B", [2, 4])
def test_multimodal_split_executes_processors(tmp_path, monkeypatch, B):
    state = _State(native=False)
    policy, pre, post, server = _compose(tmp_path, monkeypatch, state, "split_and_stack")
    try:
        processed = pre(_batch(B))
        action = policy.select_action(processed)
        final = post(action)
        assert tuple(final.shape) == (B, ACTION_DIM)
        assert state.n == B                     # split = B requests
        assert len(state.bodies[0]["observation"]["observation.images.front"]) == C
    finally:
        server.__exit__()
