# test_e2e_http_no_mocks.py — hermetic B=1 end-to-end, no mocks (v1.3.5).
#
# The whole point of this file: prove the plugin's boundaries COMPOSE on a real
# wire, not just in isolation. Nothing here is patched — not CoreAIPolicy, not
# RunnerClient, not the transport. A real local HTTP server implements the runner
# protocol; a real CoreAIBridgePolicy.from_pretrained loads a local canonical
# artifact, opens a real RunnerClient, negotiates capabilities over HTTP, POSTs a
# real observation, and the strict validator turns the response into Tensor[1, A].
#
# Deferred to v1.3.6 (documented): the full official make_policy /
# make_pre_post_processors composition. This file exercises the real
# from_pretrained -> RunnerClient -> HTTP -> negotiation -> POST -> validation
# chain end to end without mocks.

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from lerobot_coreai.errors import CoreAIPolicyError  # noqa: E402
from lerobot_policy_coreai_bridge import (  # noqa: E402
    CoreAIBridgeConfig, CoreAIBridgePolicy,
)

HORIZON, ACTION_DIM = 3, 7


def _manifest_dict():
    """A minimal but valid local CoreAI artifact manifest (chunk [3, 7], so100)."""
    return {
        "schema_version": "lerobot-coreai.v0",
        "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0", "commit": None},
        "policy": {"repo_id": "kevinqz/E2E-SO100-CoreAI",
                   "source_repo_id": "lerobot/e2e", "type": "evo1",
                   "class": None, "config_class": None},
        "robot": {"type": "so100", "action_representation": "joint_position_delta",
                  "fps": 30},
        "features": {
            "observation": {
                "observation.state": {"dtype": "float32", "shape": [ACTION_DIM],
                                      "required": True},
                "task": {"dtype": "string", "required": False},
            },
            "action": {"action": {"dtype": "float32", "shape": [HORIZON, ACTION_DIM]}},
        },
        "normalization": {"format": "lerobot", "path": "norm_stats.json",
                          "sha256": None},
        "coreai": {"artifact_format": "aimodel", "runner": "coreai-runner",
                   "graphs": [{"name": "action_denoise_step", "role": "denoise_step"}],
                   "host_loop_required": False},
        "evaluation": {"metric": "action_parity", "status": "passed", "n_obs": 8,
                       "min_chunk_cosine": 0.9999, "max_action_mae": None,
                       "max_relative_action_mae": None,
                       "proves_numeric_fidelity": True, "proves_task_success": False,
                       "proves_robot_safety": False},
        "safety": {"default_mode": "dry_run",
                   "real_actuation_requires_confirmation": True},
    }


def _write_artifact(tmp_path):
    (tmp_path / "lerobot-coreai.json").write_text(json.dumps(_manifest_dict()))
    return str(tmp_path)


class _RunnerState:
    """Mutable server config + captured wire payloads (thread-shared)."""
    def __init__(self, *, protocol="coreai-runner.v2", encodings=("nested_json_v1",),
                 action=None, capabilities_status=200, backward_compatible_with=()):
        self.protocol = protocol
        self.encodings = list(encodings)
        self.action = action or [[float(i)] * ACTION_DIM for i in range(HORIZON)]
        self.capabilities_status = capabilities_status
        self.backward_compatible_with = list(backward_compatible_with)
        self.last_predict_body = None
        self.predict_count = 0


def _make_handler(state: _RunnerState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/health":
                return self._json(200, {"status": "ok"})
            if self.path == "/v1/capabilities":
                if state.capabilities_status != 200:
                    return self._json(state.capabilities_status,
                                      {"error": {"message": "boom"}})
                return self._json(200, {
                    "runtime": "coreai-runner",
                    "supports": {"action": True},
                    "protocol_version": state.protocol,
                    "observation_encodings": state.encodings,
                    "backward_compatible_with": state.backward_compatible_with,
                    "action_batching": {"supported": False, "max_batch_size": 1},
                })
            return self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/predict":
                return self._json(404, {"error": {"message": "not found"}})
            n = int(self.headers.get("Content-Length", 0))
            state.last_predict_body = json.loads(self.rfile.read(n) or b"{}")
            state.predict_count += 1
            return self._json(200, {"action": state.action})
    return Handler


class _Server:
    def __init__(self, state: _RunnerState):
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
        self.port = self._srv.server_address[1]
        self.state = state

    def __enter__(self):
        self._t = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._srv.shutdown()
        self._srv.server_close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"


def _load_policy(tmp_path, monkeypatch, state, **cfg_kw):
    artifact = _write_artifact(tmp_path)
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    cfg = CoreAIBridgeConfig(coreai_artifact=artifact, **cfg_kw)
    policy = CoreAIBridgePolicy.from_pretrained(artifact, config=cfg)
    return policy, server


# --- Happy path: full no-mock chain -> Tensor[1, A] ---

def test_e2e_predict_produces_tensor_1xA(tmp_path, monkeypatch):
    state = _RunnerState()
    policy, server = _load_policy(
        tmp_path, monkeypatch, state,
        expected_action_dim=ACTION_DIM, expected_action_horizon=HORIZON,
        expected_robot_type="so100")
    try:
        # real RunnerClient over real HTTP; no mocks anywhere.
        assert policy.coreai_policy.runner is not None
        action = policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM)})
        assert isinstance(action, torch.Tensor)
        assert action.shape == (1, ACTION_DIM)
        assert torch.isfinite(action).all()
        assert state.predict_count == 1
    finally:
        server.__exit__()


def test_e2e_wire_payload_is_correct(tmp_path, monkeypatch):
    state = _RunnerState()
    policy, server = _load_policy(tmp_path, monkeypatch, state)
    try:
        policy.select_action({
            "observation.state": torch.arange(ACTION_DIM, dtype=torch.float32
                                              ).reshape(1, ACTION_DIM),
            "task": ["pick up the cube"],
            # label-leakage bait — must NOT reach the runner:
            "action": torch.zeros(1, ACTION_DIM),
            "index": torch.zeros(1),
            "reward": torch.zeros(1),
            "timestamp": torch.zeros(1),
        })
        body = state.last_predict_body
        obs = body["observation"]
        # batch dim stripped; state is a plain JSON list of length A.
        assert obs["observation.state"] == [float(i) for i in range(ACTION_DIM)]
        assert obs["task"] == "pick up the cube"          # list[str] -> str
        for leaked in ("action", "index", "reward", "timestamp"):
            assert leaked not in obs
        # negotiated protocol/encoding + observation hash ride in options.
        opts = body["options"]
        assert opts["observation_encoding"] == "nested_json_v1"
        assert opts["protocol_version"] == "coreai-runner.v2"
        assert opts["observation_sha256"].startswith("sha256:")
        # no secret / url / token persisted in the wire payload.
        blob = json.dumps(body)
        assert server.url not in blob and "token" not in blob.lower()
    finally:
        server.__exit__()


def test_e2e_negotiated_protocol_uses_announced_version(tmp_path, monkeypatch):
    # runner announces v3 (declaring back-compat) -> request carries v3, not v2.
    state = _RunnerState(protocol="coreai-runner.v3",
                         backward_compatible_with=("coreai-runner.v2",))
    policy, server = _load_policy(tmp_path, monkeypatch, state)
    try:
        policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM)})
        assert state.last_predict_body["options"]["protocol_version"] == \
            "coreai-runner.v3"
    finally:
        server.__exit__()


# --- Failure paths: fail closed, never silent legacy ---

def test_e2e_lower_protocol_fails_closed(tmp_path, monkeypatch):
    state = _RunnerState(protocol="coreai-runner.v1")
    policy, server = _load_policy(tmp_path, monkeypatch, state)
    try:
        with pytest.raises(CoreAIPolicyError):
            policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM)})
        assert state.predict_count == 0     # never reached the runner
    finally:
        server.__exit__()


def test_e2e_no_common_encoding_fails_closed(tmp_path, monkeypatch):
    state = _RunnerState(encodings=("some_future_v9",))
    policy, server = _load_policy(tmp_path, monkeypatch, state)
    try:
        with pytest.raises(CoreAIPolicyError):
            policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM)})
        assert state.predict_count == 0
    finally:
        server.__exit__()


def test_e2e_wrong_horizon_response_fails_closed(tmp_path, monkeypatch):
    # runner returns 8 rows though the manifest declares horizon 3.
    state = _RunnerState(action=[[0.0] * ACTION_DIM for _ in range(8)])
    policy, server = _load_policy(tmp_path, monkeypatch, state)
    try:
        with pytest.raises(Exception):
            policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM)})
    finally:
        server.__exit__()


def test_e2e_missing_protocol_without_legacy_fails(tmp_path, monkeypatch):
    state = _RunnerState(protocol=None)
    policy, server = _load_policy(tmp_path, monkeypatch, state)  # legacy not allowed
    try:
        with pytest.raises(CoreAIPolicyError):
            policy.select_action({"observation.state": torch.zeros(1, ACTION_DIM)})
        assert state.predict_count == 0
    finally:
        server.__exit__()


def test_e2e_missing_protocol_with_legacy_opt_in_succeeds(tmp_path, monkeypatch):
    state = _RunnerState(protocol=None)
    policy, server = _load_policy(tmp_path, monkeypatch, state,
                                  runtime_binding_mode="legacy")
    try:
        with pytest.warns(RuntimeWarning):
            action = policy.select_action(
                {"observation.state": torch.zeros(1, ACTION_DIM)})
        assert action.shape == (1, ACTION_DIM)
        assert state.last_predict_body["options"]["protocol_version"] == \
            "coreai-runner.v2"      # minimum, used as the legacy label
    finally:
        server.__exit__()
