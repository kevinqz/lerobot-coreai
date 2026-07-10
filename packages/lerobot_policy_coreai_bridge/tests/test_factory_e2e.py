# test_factory_e2e.py — full official-factory composition, no mocks (v1.3.6).
#
# This is the capstone: a canonical artifact traverses the ENTIRE official path
#   register_third_party_plugins
#     -> PreTrainedConfig.from_pretrained(artifact)
#     -> make_policy(cfg, ds_meta=...)                     (real factory)
#     -> make_pre_post_processors(cfg, pretrained_path=..) (real processors, from JSON)
#     -> post(policy.select_action(pre(batch)))            (real CoreAIPolicy + RunnerClient)
#     -> Tensor[1, A]
# against a REAL local HTTP runner. Nothing is patched: not CoreAIPolicy, not
# RunnerClient, not make_policy, not the processors. Proves protocol/factory
# composition only — NOT official lerobot-eval, task success, or physical safety.

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


def _manifest(action_dim=ACTION_DIM):
    return {
        "schema_version": "lerobot-coreai.v0", "runtime": "coreai",
        "framework": {"name": "lerobot", "version": "0.6.0", "commit": None},
        "policy": {"repo_id": "kevinqz/E2E", "source_repo_id": "lerobot/e2e",
                   "type": "evo1", "class": None, "config_class": None},
        "robot": {"type": "so100", "action_representation": "joint_position_delta",
                  "fps": 30},
        "features": {
            "observation": {
                "observation.state": {"dtype": "float32", "shape": [action_dim],
                                      "required": True},
                "task": {"dtype": "string", "required": False}},
            "action": {"action": {"dtype": "float32", "shape": [HORIZON, action_dim]}}},
        "normalization": {"format": "lerobot", "path": "norm_stats.json",
                          "sha256": None},
        "coreai": {"artifact_format": "aimodel", "runner": "coreai-runner",
                   "graphs": [{"name": "g", "role": "denoise_step"}],
                   "host_loop_required": False},
        "evaluation": {"metric": "action_parity", "status": "passed", "n_obs": 8,
                       "min_chunk_cosine": 0.9999, "max_action_mae": None,
                       "max_relative_action_mae": None, "proves_numeric_fidelity": True,
                       "proves_task_success": False, "proves_robot_safety": False},
        "safety": {"default_mode": "dry_run",
                   "real_actuation_requires_confirmation": True},
        "contracts": {"processor": {
            "observation_input": {"owner": "coreai_runner", "expects": "raw"},
            "action_output": {"owner": "coreai_runner", "returns": "post"}}},
    }


class _State:
    def __init__(self):
        self.body = None
        self.n = 0
        self.action = [[float(i)] * ACTION_DIM for i in range(HORIZON)]


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
                return self._j(200, {
                    "runtime": "coreai-runner", "supports": {"action": True},
                    "protocol_version": "coreai-runner.v2",
                    "observation_encodings": ["nested_json_v1"],
                    "action_batching": {"supported": False, "max_batch_size": 1}})
            return self._j(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            state.body = json.loads(self.rfile.read(n) or b"{}")
            state.n += 1
            return self._j(200, {"action": state.action})
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


def _artifact(tmp_path, action_dim=ACTION_DIM):
    src = tmp_path / "coreai"
    src.mkdir()
    (src / "lerobot-coreai.json").write_text(json.dumps(_manifest(action_dim)))
    art = tmp_path / "plugin"
    build_plugin_artifact(str(src), str(art), runner_url_env="COREAI_RUNNER_URL")
    return str(art)


def _ds_meta(action_dim=ACTION_DIM):
    # Minimal LeRobotDatasetMetadata stand-in: make_policy only needs .features
    # (to derive input/output PolicyFeatures) and .stats.
    return types.SimpleNamespace(
        features={
            "observation.state": {"dtype": "float32", "shape": [action_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
        },
        stats={})


def _compose(art, monkeypatch, state, action_dim=ACTION_DIM):
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(art)
    cfg.pretrained_path = art
    policy = make_policy(cfg, ds_meta=_ds_meta(action_dim))         # real factory
    pre, post = make_pre_post_processors(cfg, pretrained_path=art)  # real processors
    return policy, pre, post, server


def test_official_factory_composition_yields_tensor_1xA(tmp_path, monkeypatch):
    art = _artifact(tmp_path)
    state = _State()
    policy, pre, post, server = _compose(art, monkeypatch, state)
    try:
        batch = {"observation.state": torch.zeros(1, ACTION_DIM),
                 "task": ["pick up cube"]}
        final = post(policy.select_action(pre(batch)))
        assert isinstance(final, torch.Tensor)
        assert tuple(final.shape) == (1, ACTION_DIM)
        assert state.n == 1                                  # exactly one predict
        # config.input_features / output_features were populated by make_policy.
        assert "observation.state" in policy.config.input_features
        assert "action" in policy.config.output_features
    finally:
        server.__exit__()


def test_official_composition_no_label_leakage_on_wire(tmp_path, monkeypatch):
    art = _artifact(tmp_path)
    state = _State()
    policy, pre, post, server = _compose(art, monkeypatch, state)
    try:
        batch = {"observation.state": torch.arange(ACTION_DIM, dtype=torch.float32
                                                   ).reshape(1, ACTION_DIM),
                 "task": ["pick up cube"]}
        # pre() injects action/next.* placeholders — the transport must drop them.
        post(policy.select_action(pre(batch)))
        obs = state.body["observation"]
        assert set(obs.keys()) == {"observation.state", "task"}
        assert obs["task"] == "pick up cube"
        opts = state.body["options"]
        assert opts["protocol_version"] == "coreai-runner.v2"
        assert opts["observation_encoding"] == "nested_json_v1"
        assert opts["observation_sha256"].startswith("sha256:")
        assert server.url not in json.dumps(state.body)
    finally:
        server.__exit__()


def test_official_composition_processors_loaded_from_disk(tmp_path, monkeypatch):
    art = _artifact(tmp_path)
    state = _State()
    policy, pre, post, server = _compose(art, monkeypatch, state)
    try:
        # real PolicyProcessorPipeline reconstructed from the artifact JSON files.
        assert type(pre).__name__ == "DataProcessorPipeline"
        assert type(post).__name__ == "DataProcessorPipeline"
        assert list(pre.steps) == [] and list(post.steps) == []
    finally:
        server.__exit__()


def test_official_composition_feature_mismatch_fails(tmp_path, monkeypatch):
    # Artifact declares action_dim 7; the dataset presents action_dim 9 -> the
    # feature cross-binding in from_pretrained (via make_policy) must fail closed.
    art = _artifact(tmp_path, action_dim=ACTION_DIM)
    state = _State()
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    register_third_party_plugins()
    cfg = PreTrainedConfig.from_pretrained(art)
    cfg.pretrained_path = art
    try:
        with pytest.raises(Exception):
            make_policy(cfg, ds_meta=_ds_meta(action_dim=9))
    finally:
        server.__exit__()
