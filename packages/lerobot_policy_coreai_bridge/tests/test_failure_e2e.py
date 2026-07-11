# test_failure_e2e.py — v1.3.21 failure paths through the REAL policy/transport
# /negotiation stack (P1.8/K, partial). No mocks: a real local HTTP runner is
# configured to fail at a specific stage; the policy emits RUNTIME-origin terminal
# events (execution.failed), a FailureEvidence v2 bundle is written and the OFFLINE
# verifier re-proves it. Runs wherever lerobot+torch are installed (stable/dev jobs).
#
# NOTE (v1.3.21): these drive the policy/transport/negotiation stack directly rather
# than through lerobot_eval.rollout; wiring the injected failures into the official
# rollout CI matrix is the remaining step (v1.3.22).

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lerobot")
import torch  # noqa: E402

from lerobot_coreai.rollout_verify import verify_official_rollout_evidence  # noqa: E402
from lerobot_policy_coreai_bridge import (  # noqa: E402
    CoreAIBridgeConfig, CoreAIBridgePolicy,
)
from lerobot_policy_coreai_bridge.rollout_evidence import (  # noqa: E402
    capture_environment_identity, write_failure_evidence,
)

HORIZON, A = 3, 7


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
                   "real_actuation_requires_confirmation": True}}


class _State:
    def __init__(self, *, protocol="coreai-runner.v2", caps_status=200,
                 predict_status=200, action=None):
        self.protocol = protocol
        self.caps_status = caps_status
        self.predict_status = predict_status
        self.action = action or [[float(i)] * A for i in range(HORIZON)]


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
                if state.caps_status != 200:
                    return self._j(state.caps_status, {"error": {"message": "boom"}})
                return self._j(200, {"runtime": "coreai-runner",
                                     "supports": {"action": True},
                                     "protocol_version": state.protocol,
                                     "observation_encodings": ["nested_json_v1"],
                                     "action_batching": {"supported": False,
                                                         "max_batch_size": 1}})
            return self._j(404, {})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            self.rfile.read(n)
            if state.predict_status != 200:
                return self._j(state.predict_status, {"error": {"message": "boom"}})
            return self._j(200, {"action": state.action})
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


def _policy(tmp_path, monkeypatch, state):
    (tmp_path / "lerobot-coreai.json").write_text(json.dumps(_manifest()))
    server = _Server(state).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    cfg = CoreAIBridgeConfig(coreai_artifact=str(tmp_path))
    policy = CoreAIBridgePolicy.from_pretrained(str(tmp_path), config=cfg)
    return policy, server


def _drive_failure(tmp_path, monkeypatch, state, failed_stage, case):
    """Run the real stack until it fails, seal a runtime-origin failure bundle,
    and re-prove it offline. Returns the verify result."""
    policy, server = _policy(tmp_path, monkeypatch, state)
    ev_dir = tmp_path / "ev"
    try:
        policy.begin_evidence_session(f"failed-{case}")
        with pytest.raises(Exception):  # noqa: PT011 — any real stage failure
            policy.select_action({"observation.state": torch.zeros(1, A)})
        events = policy.abort_evidence_session(failed_stage, detail="injected")
        assert events[-1]["event"] == "execution.failed"      # RUNTIME terminal
        write_failure_evidence(
            str(ev_dir / case), case=case, failed_stage=failed_stage,
            exception_type="RuntimeError", message="injected failure",
            target="local", execution_id=f"failed-{case}",
            environment=capture_environment_identity("local"),
            partial_events=tuple(events), terminal_event_origin="runtime")
    finally:
        server.__exit__()
    return verify_official_rollout_evidence(str(ev_dir), require_complete_matrix=False)


def test_negotiation_failure_produces_verifiable_bundle(tmp_path, monkeypatch):
    # runner announces a protocol below the minimum -> negotiation fails.
    res = _drive_failure(tmp_path, monkeypatch, _State(protocol="coreai-runner.v1"),
                         "runner_negotiate", "failure-negotiation-b1")
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}


def test_bind_time_capabilities_failure_produces_writer_synthesized_bundle(
        tmp_path, monkeypatch):
    # a capabilities HTTP 500 fails at BIND (factory_load), before any evidence
    # session — so the terminal event is honestly writer-synthesized (diagnostic).
    (tmp_path / "lerobot-coreai.json").write_text(json.dumps(_manifest()))
    server = _Server(_State(caps_status=500)).__enter__()
    monkeypatch.setenv("COREAI_RUNNER_URL", server.url)
    ev_dir = tmp_path / "ev"
    try:
        cfg = CoreAIBridgeConfig(coreai_artifact=str(tmp_path))
        with pytest.raises(Exception):  # noqa: PT011
            CoreAIBridgePolicy.from_pretrained(str(tmp_path), config=cfg)
        write_failure_evidence(
            str(ev_dir / "failure-bind-b1"), case="failure-bind-b1",
            failed_stage="factory_load", exception_type="RunnerExecutionError",
            message="capabilities HTTP 500 at bind", target="local",
            environment=capture_environment_identity("local"))
    finally:
        server.__exit__()
    res = verify_official_rollout_evidence(str(ev_dir), require_complete_matrix=False)
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}
    rep = json.loads((ev_dir / "failure-bind-b1" / "failure_report.json").read_text())
    assert rep["terminal_event_origin"] == "writer_synthesized"


def test_predict_http_failure_produces_verifiable_bundle(tmp_path, monkeypatch):
    res = _drive_failure(tmp_path, monkeypatch, _State(predict_status=500),
                         "request", "failure-predict-http-b1")
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}


def test_nonfinite_action_produces_verifiable_bundle(tmp_path, monkeypatch):
    bad = [[float("inf")] * A for _ in range(HORIZON)]
    res = _drive_failure(tmp_path, monkeypatch, _State(action=bad),
                         "validation", "failure-nonfinite-b1")
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}


def test_malformed_response_shape_produces_verifiable_bundle(tmp_path, monkeypatch):
    wrong = [[float(i)] * A for i in range(HORIZON + 2)]     # wrong horizon
    res = _drive_failure(tmp_path, monkeypatch, _State(action=wrong),
                         "validation", "failure-malformed-b1")
    assert res.ok, {k: v for k, v in res.checks.items() if v != "passed"}


def test_synthesized_terminal_is_labeled_writer_synthesized(tmp_path):
    # a setup-stage failure with no runtime trace -> writer synthesizes the terminal
    # event, and the report must honestly label it as such (never runtime).
    write_failure_evidence(
        str(tmp_path / "failure-setup-b1"), case="failure-setup-b1",
        failed_stage="setup", exception_type="OSError", message="no artifact",
        target="local")
    rep = json.loads((tmp_path / "failure-setup-b1" / "failure_report.json").read_text())
    assert rep["terminal_event_origin"] == "writer_synthesized"
    assert verify_official_rollout_evidence(
        str(tmp_path), require_complete_matrix=False).ok
